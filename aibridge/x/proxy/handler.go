package proxy

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/xerrors"

	"cdr.dev/slog/v3"
	"github.com/coder/coder/v2/aibridge/circuitbreaker"
	aibclient "github.com/coder/coder/v2/aibridge/client"
	aibcontext "github.com/coder/coder/v2/aibridge/context"
	"github.com/coder/coder/v2/aibridge/credential"
	"github.com/coder/coder/v2/aibridge/headers"
	"github.com/coder/coder/v2/aibridge/intercept/apidump"
	"github.com/coder/coder/v2/aibridge/interceptionerror"
	"github.com/coder/coder/v2/aibridge/keypool"
	"github.com/coder/coder/v2/aibridge/metrics"
	"github.com/coder/coder/v2/aibridge/provider"
	"github.com/coder/coder/v2/aibridge/recorder"
	"github.com/coder/coder/v2/aibridge/routing"
	"github.com/coder/coder/v2/aibridge/tracing"
	"github.com/coder/coder/v2/aibridge/utils"
	"github.com/coder/quartz"
)

type forwardingHandler struct {
	provider  provider.Provider
	logger    slog.Logger
	metrics   *metrics.Metrics
	tracer    trace.Tracer
	transport http.RoundTripper
	failover  keypool.KeyFailoverConfig
	breaker   *circuitbreaker.ProviderCircuitBreakers
	proxy     *httputil.ReverseProxy
}

// newForwardingHandler constructs one forwarding engine per provider snapshot.
// The router owns the returned transport and closes idle connections on retirement.
func newForwardingHandler(prov provider.Provider, logger slog.Logger, m *metrics.Metrics, tracer trace.Tracer) (*forwardingHandler, *http.Transport, error) {
	baseURL, err := url.Parse(prov.BaseURL())
	if err != nil {
		return nil, nil, xerrors.Errorf("configure provider %q base URL: %w", prov.Name(), err)
	}
	transport := utils.NewStreamingTransport()
	transport.DisableCompression = true
	h := &forwardingHandler{
		provider:  prov,
		logger:    logger,
		metrics:   m,
		tracer:    tracer,
		transport: apidump.NewPassthroughMiddleware(transport, prov.APIDumpDir(), prov.Name(), logger, quartz.NewReal()),
		failover:  prov.KeyFailoverConfig(logger),
		breaker:   circuitbreaker.NewProviderCircuitBreakers(prov.Name(), prov.CircuitBreakerConfig(), logger, m),
	}
	h.proxy = &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			rewriteForwardingURL(pr, prov.RoutePrefix(), baseURL)
			pr.Out.Header = prepareForwardingHeaders(pr)
		},
		Transport:     h,
		FlushInterval: -1,
		ErrorHandler: func(_ http.ResponseWriter, r *http.Request, err error) {
			observationFromContext(r.Context()).err = err
		},
		ErrorLog: slog.Stdlib(context.Background(), logger.With(slog.F("provider", prov.Name())), slog.LevelWarn),
	}
	return h, transport, nil
}

func rewriteForwardingURL(pr *httputil.ProxyRequest, routePrefix string, baseURL *url.URL) {
	pr.Out.URL.Path = strings.TrimPrefix(pr.In.URL.Path, routePrefix)
	if pr.In.URL.RawPath != "" {
		pr.Out.URL.RawPath = strings.TrimPrefix(pr.In.URL.RawPath, routePrefix)
	}
	// ReverseProxy may sanitize query parameters; forwarding preserves the bytes.
	pr.Out.URL.RawQuery = pr.In.URL.RawQuery
	pr.SetURL(baseURL)
}

func prepareForwardingHeaders(pr *httputil.ProxyRequest) http.Header {
	prepared := headers.PrepareClientHeaders(pr.Out.Header)
	// Unlike SDK requests, forwarding preserves the client's encoding choice.
	if values, ok := pr.Out.Header["Accept-Encoding"]; ok {
		prepared["Accept-Encoding"] = values
	}
	// In contains only the selected credential. Restore it even if Connection
	// nominated it for removal by ReverseProxy.
	for _, name := range []string{headers.AuthHeaderAuthorization, headers.AuthHeaderXAPIKey} {
		if value := pr.In.Header.Get(name); value != "" {
			prepared.Set(name, value)
		}
	}
	return prepared
}

func (h *forwardingHandler) serve(processCtx context.Context, w http.ResponseWriter, r *http.Request, rec recorder.Recorder, tasks *utils.ConcurrentGroup) {
	start := time.Now()
	ctx, span := h.tracer.Start(r.Context(), "Proxy")
	defer span.End()
	record, cred := h.checkRequest(w, r, rec, tasks)
	if record == nil {
		return
	}
	record.ID, record.StartedAt = uuid.NewString(), start.UTC()
	fields := []slog.Field{slog.F("interception_id", record.ID), slog.F("provider", h.provider.Name()), slog.F("credential_kind", string(cred.Kind()))}
	ctx = slog.With(ctx, fields...)
	processCtx = slog.With(trace.ContextWithSpan(processCtx, span), fields...)
	span.SetAttributes(attribute.String(tracing.InterceptionID, record.ID), attribute.String(tracing.InitiatorID, record.InitiatorID), attribute.String(tracing.Provider, h.provider.Name()))
	state := &responseObservation{credentialHint: record.CredentialHint}
	forward, body := h.prepareForwarding(r.WithContext(context.WithValue(ctx, observationContextKey{}, state)), cred)
	started := make(chan error, 1)
	finished := make(chan recorder.InterceptionRecordEnded, 1)
	request := r.Clone(processCtx)
	request.Body, request.GetBody = nil, nil
	tasks.Go(func() error { return h.processRequest(processCtx, request, rec, *record, body, started) })
	tasks.Go(func() error { return h.processResponse(processCtx, rec, started, finished) })

	route := strings.TrimPrefix(r.Pattern, "/"+h.provider.Name())
	if h.metrics != nil {
		h.metrics.InterceptionsInflight.WithLabelValues(h.provider.Name(), "", route).Inc()
	}
	defer func() {
		panicValue := recover()
		h.finishForwarding(ctx, r, start, record, body, state, finished, panicValue)
		if panicValue != nil {
			panic(panicValue)
		}
	}()
	state.err = h.breaker.Execute(route, "", w, func(rw http.ResponseWriter) error {
		writer := &errorCapturingWriter{ResponseWriter: rw, client: w}
		defer func() {
			if state.err == nil {
				state.err = writer.Error()
			}
			if state.body != nil && !state.body.closed {
				_ = state.body.Close()
			}
		}()
		h.proxy.ServeHTTP(writer, forward)
		if state.body == nil && state.err != nil {
			if _, ok := errors.AsType[*http.MaxBytesError](state.err); ok {
				routing.WriteRequestBodyTooLarge(ctx, writer)
			} else {
				h.logger.Warn(ctx, "upstream proxy error", slog.Error(state.err))
				http.Error(writer, "upstream proxy error", http.StatusBadGateway)
			}
		}
		if writer.Error() != nil || (state.body != nil && !state.body.eof) {
			if state.err == nil {
				state.err = writer.Error()
			}
			panic(http.ErrAbortHandler)
		}
		return state.err
	})
}

func (h *forwardingHandler) checkRequest(w http.ResponseWriter, r *http.Request, rec recorder.Recorder, tasks *utils.ConcurrentGroup) (*recorder.InterceptionRecord, credential.Credential) {
	if headers.IsWebSocketUpgrade(r) {
		http.Error(w, "WebSocket transport is not supported, use HTTP", http.StatusNotImplemented)
		return nil, nil
	}
	firewallID, firewallSequence, err := headers.ExtractAgentFirewallHeaders(r)
	if err != nil {
		http.Error(w, "invalid agent firewall headers", http.StatusBadRequest)
		return nil, nil
	}
	if r.ContentLength > routing.MaxRequestBodyBytes {
		if r.Body != nil {
			_ = r.Body.Close()
		}
		routing.WriteRequestBodyTooLarge(r.Context(), w)
		return nil, nil
	}
	actor := aibcontext.ActorFromContext(r.Context())
	if actor == nil {
		http.Error(w, "no actor found", http.StatusBadRequest)
		return nil, nil
	}
	if rec == nil || tasks == nil {
		http.Error(w, "recorder unavailable", http.StatusInternalServerError)
		return nil, nil
	}
	cred, err := h.provider.ResolveCredential(r)
	if err != nil {
		http.Error(w, "upstream authentication unavailable", http.StatusBadGateway)
		return nil, nil
	}
	switch cred.(type) {
	case credential.BYOK, *credential.CentralizedPool:
	default:
		http.Error(w, "upstream authentication is not supported in proxy mode", http.StatusNotImplemented)
		return nil, nil
	}
	return &recorder.InterceptionRecord{
		InitiatorID:                 actor.ID,
		Metadata:                    actor.Metadata,
		Provider:                    h.provider.Type(),
		ProviderName:                h.provider.Name(),
		Client:                      string(aibclient.GuessClient(r)),
		UserAgent:                   r.UserAgent(),
		AgentFirewallSessionID:      firewallID,
		AgentFirewallSequenceNumber: firewallSequence,
		CredentialKind:              cred.Kind(),
		CredentialHint:              cred.Hint(),
	}, cred
}

func (*forwardingHandler) prepareForwarding(r *http.Request, cred credential.Credential) (*http.Request, *requestBuffer) {
	body := &requestBuffer{source: r.Body, captured: make(chan capturedBody, 1)}
	if r.Body == nil || r.Body == http.NoBody {
		body.err = io.EOF
		body.finishCapture()
	}
	r = r.Clone(r.Context())
	if r.Body != nil && r.Body != http.NoBody && r.ContentLength == 0 {
		r.ContentLength = -1 // In-process callers may use zero for an unknown length.
	}
	r.Body, _ = body.getBody()
	r.GetBody = body.getBody
	r.Header.Del(headers.AuthHeaderAuthorization)
	r.Header.Del(headers.AuthHeaderXAPIKey)
	if byok, ok := credential.AsBYOK(cred); ok {
		value := byok.Secret
		if byok.Header == headers.AuthHeaderAuthorization {
			value = "Bearer " + value
		}
		r.Header.Set(byok.Header, value)
	}
	return r, body
}

func (h *forwardingHandler) processRequest(ctx context.Context, r *http.Request, rec recorder.Recorder, record recorder.InterceptionRecord, body *requestBuffer, started chan<- error) (outErr error) {
	defer func() {
		if recover() != nil {
			outErr = xerrors.New("request processing panicked")
		}
		started <- outErr
	}()
	select {
	case capture := <-body.captured:
		if capture.err == nil {
			record.ClientSessionID = aibclient.GuessSessionIDFromPayload(aibclient.Type(record.Client), r, capture.payload)
		} else {
			h.logger.Debug(ctx, "request body metadata unavailable", slog.Error(capture.err))
		}
	case <-ctx.Done():
		return ctx.Err()
	}
	recordCtx, cancel := context.WithTimeout(ctx, recorder.DefaultAsyncTimeout)
	defer cancel()
	if err := rec.RecordInterception(recordCtx, &record); err != nil {
		return xerrors.Errorf("record interception: %w", err)
	}
	return nil
}

func (*forwardingHandler) processResponse(ctx context.Context, rec recorder.Recorder, started <-chan error, finished <-chan recorder.InterceptionRecordEnded) (outErr error) {
	defer func() {
		if recover() != nil {
			outErr = xerrors.New("response processing panicked")
		}
	}()
	var record recorder.InterceptionRecordEnded
	select {
	case record = <-finished:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-started:
		if err != nil {
			return nil // The start job reports its error; there is no row to update.
		}
	case <-ctx.Done():
		return ctx.Err()
	}
	recordCtx, cancel := context.WithTimeout(ctx, recorder.DefaultAsyncTimeout)
	defer cancel()
	if err := rec.RecordInterceptionEnded(recordCtx, &record); err != nil {
		return xerrors.Errorf("record interception end: %w", err)
	}
	return nil
}

// finishForwarding publishes an immutable outcome without waiting for processing.
func (h *forwardingHandler) finishForwarding(ctx context.Context, r *http.Request, start time.Time, record *recorder.InterceptionRecord, body *requestBuffer, state *responseObservation, finished chan<- recorder.InterceptionRecordEnded, panicValue any) {
	end := time.Now()
	body.finishCapture()
	terminal := state.err
	body.mu.Lock()
	requestErr := body.err
	body.mu.Unlock()
	if terminal == nil && state.status < http.StatusBadRequest && !errors.Is(requestErr, io.EOF) {
		terminal = requestErr
	}
	if terminal == nil && ctx.Err() != nil && (state.body == nil || !state.body.eof) {
		terminal = ctx.Err()
	}
	if terminal == nil && state.body != nil {
		terminal = state.body.readErr
	}
	if terminal == nil && panicValue != nil {
		terminal = xerrors.New("response stream aborted")
	}
	errType, message := interceptionerror.Categorize(h.provider, terminal)
	if _, ok := errors.AsType[*http.MaxBytesError](terminal); ok {
		errType = recorder.ErrorTypeBadRequest
	}
	if terminal == nil && state.status >= http.StatusBadRequest {
		errType, message = recorder.ErrorTypeFromStatus(state.status), http.StatusText(state.status)
	}
	status := metrics.InterceptionCountStatusCompleted
	if errType != "" {
		status = metrics.InterceptionCountStatusFailed
		trace.SpanFromContext(ctx).SetStatus(codes.Error, message)
	}
	if h.metrics != nil {
		route := strings.TrimPrefix(r.Pattern, "/"+h.provider.Name())
		h.metrics.InterceptionsInflight.WithLabelValues(h.provider.Name(), "", route).Dec()
		h.metrics.InterceptionDuration.WithLabelValues(h.provider.Name(), "").Observe(end.Sub(start).Seconds())
		h.metrics.InterceptionCount.WithLabelValues(h.provider.Name(), "", status, route, r.Method, record.InitiatorID, record.Client).Inc()
	}
	finished <- recorder.InterceptionRecordEnded{ID: record.ID, EndedAt: end.UTC(), CredentialHint: state.credentialHint, ErrorType: errType, ErrorMessage: message}
}

// RoundTrip observes only the final response returned by key failover. Its
// callbacks and bookkeeping belong to this request, not the shared proxy.
func (h *forwardingHandler) RoundTrip(r *http.Request) (*http.Response, error) {
	state := observationFromContext(r.Context())
	cfg := h.failover
	if cfg.Pool != nil {
		inject, build := cfg.InjectAuthKey, cfg.BuildKeyPoolResponse
		cfg.InjectAuthKey = func(header *http.Header, key string) {
			state.credentialHint = utils.MaskSecret(key)
			inject(header, key)
		}
		cfg.BuildKeyPoolResponse = func(err *keypool.Error) *http.Response {
			state.err = err
			return build(err)
		}
	}
	resp, err := keypool.NewKeyFailoverTransport(h.transport, cfg).RoundTrip(r)
	if err != nil {
		return resp, err
	}
	if resp.StatusCode == http.StatusSwitchingProtocols {
		_ = resp.Body.Close()
		return nil, xerrors.New("upstream protocol upgrades are not supported")
	}
	state.status = resp.StatusCode
	state.body = &observedBody{ReadCloser: resp.Body}
	resp.Body = state.body
	return resp, nil
}

// requestBuffer gives each reader its own position in the captured input.
// A reader at the buffered end pulls more input; others can still read the prefix.
type requestBuffer struct {
	source  io.Reader
	readMu  sync.Mutex // Serializes reads from source, not reads of buffered bytes.
	mu      sync.Mutex // Protects payload and err.
	payload bytes.Buffer
	err     error

	captured    chan capturedBody
	captureSent bool // Protected by mu; notification is independent of source reads.
}

type capturedBody struct {
	payload []byte
	err     error
}

// finishCapture notifies the observer without reading or waiting for the source.
// Incomplete capture abandons body metadata, not the forwarding readers.
func (b *requestBuffer) finishCapture() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.captured == nil || b.captureSent {
		return
	}
	capture := capturedBody{err: b.err}
	switch {
	case errors.Is(b.err, io.EOF):
		capture.payload, capture.err = b.payload.Bytes(), nil
	case b.err == nil:
		capture.err = io.ErrUnexpectedEOF
	}
	b.captured <- capture
	b.captureSent = true
}

func (b *requestBuffer) getBody() (io.ReadCloser, error) {
	b.mu.Lock()
	empty := b.payload.Len() == 0 && errors.Is(b.err, io.EOF)
	b.mu.Unlock()
	if empty {
		return http.NoBody, nil
	}
	// The server owns the input body; closing one reader must not close it.
	return io.NopCloser(&requestBufferReader{buffer: b}), nil
}

type requestBufferReader struct {
	buffer *requestBuffer
	offset int
}

func (r *requestBufferReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	b := r.buffer
	n, err := b.readBuffered(p, r.offset)
	if n == 0 && err == nil {
		b.readMu.Lock()
		defer b.readMu.Unlock()
		// Another reader may have extended the buffer while we waited.
		n, err = b.readBuffered(p, r.offset)
		if n == 0 && err == nil {
			n, err = io.TeeReader(b.source, b).Read(p)
			b.mu.Lock()
			b.err = err
			b.mu.Unlock()
			if err != nil {
				b.finishCapture()
			}
		}
	}
	r.offset += n
	return n, err
}

func (b *requestBuffer) readBuffered(p []byte, offset int) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := copy(p, b.payload.Bytes()[offset:])
	if offset+n == b.payload.Len() {
		return n, b.err
	}
	return n, nil
}

func (b *requestBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.payload.Write(p)
}

// bytes fills the buffer through EOF and returns a copy of the complete payload.
func (b *requestBuffer) bytes() ([]byte, error) {
	if _, err := io.Copy(io.Discard, &requestBufferReader{buffer: b}); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return bytes.Clone(b.payload.Bytes()), nil
}

type errorCapturingWriter struct {
	http.ResponseWriter
	client http.ResponseWriter
	// ReverseProxy's initial flush can run on its timer goroutine.
	err atomic.Pointer[error]
}

func (w *errorCapturingWriter) WriteHeader(status int) {
	if status >= 100 && status < 200 {
		// The existing breaker tracks the first status. Informational responses
		// must reach the client without hiding the final status from the breaker.
		w.client.WriteHeader(status)
		return
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *errorCapturingWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	if err == nil && n < len(p) {
		err = io.ErrShortWrite
	}
	w.recordError(err)
	return n, err
}

func (w *errorCapturingWriter) recordError(err error) {
	if err != nil {
		w.err.CompareAndSwap(nil, &err)
	}
}

func (w *errorCapturingWriter) Error() error {
	if err := w.err.Load(); err != nil {
		return *err
	}
	return nil
}

func (w *errorCapturingWriter) FlushError() error {
	// The breaker wrapper only exposes Flush, which discards flush errors.
	err := http.NewResponseController(w.client).Flush()
	w.recordError(err)
	return err
}

func (w *errorCapturingWriter) Flush() { _ = w.FlushError() }
func (w *errorCapturingWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return http.NewResponseController(w.ResponseWriter).Hijack()
}
func (w *errorCapturingWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
