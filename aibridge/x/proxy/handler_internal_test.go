package proxy

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	promtest "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace/noop"
	"golang.org/x/xerrors"

	"github.com/coder/coder/v2/aibridge/circuitbreaker"
	"github.com/coder/coder/v2/aibridge/client"
	"github.com/coder/coder/v2/aibridge/config"
	aibcontext "github.com/coder/coder/v2/aibridge/context"
	"github.com/coder/coder/v2/aibridge/credential"
	"github.com/coder/coder/v2/aibridge/intercept/apidump"
	"github.com/coder/coder/v2/aibridge/keypool"
	"github.com/coder/coder/v2/aibridge/metrics"
	"github.com/coder/coder/v2/aibridge/provider"
	"github.com/coder/coder/v2/aibridge/recorder"
	"github.com/coder/coder/v2/aibridge/routing"
	"github.com/coder/coder/v2/aibridge/utils"
	codertestutil "github.com/coder/coder/v2/testutil"
	"github.com/coder/quartz"
)

// Embedding a nil Recorder makes any unexpected content recording fail loudly.
type lifecycleRecorder struct {
	recorder.Recorder
	starts           []recorder.InterceptionRecord
	ends             []recorder.InterceptionRecordEnded
	startErr, endErr error
	onStart, onEnd   func(context.Context)
}

func (r *lifecycleRecorder) RecordInterception(ctx context.Context, record *recorder.InterceptionRecord) error {
	r.starts = append(r.starts, *record)
	if r.onStart != nil {
		r.onStart(ctx)
	}
	return r.startErr
}

func (r *lifecycleRecorder) RecordInterceptionEnded(ctx context.Context, record *recorder.InterceptionRecordEnded) error {
	r.ends = append(r.ends, *record)
	if r.onEnd != nil {
		r.onEnd(ctx)
	}
	return r.endErr
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(p []byte) (int, error) { return f(p) }

type trackedBody struct {
	io.Reader
	closes, eofs int
}

func (b *trackedBody) Read(p []byte) (int, error) {
	n, err := b.Reader.Read(p)
	if err == io.EOF {
		b.eofs++
	}
	return n, err
}
func (b *trackedBody) Close() error { b.closes++; return nil }

type failingWriter struct{ http.ResponseWriter }

func (*failingWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func (w *failingWriter) Flush()                  { _ = http.NewResponseController(w.ResponseWriter).Flush() }
func (w *failingWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return http.NewResponseController(w.ResponseWriter).Hijack()
}

type failingFlushWriter struct{ http.ResponseWriter }

func (*failingFlushWriter) FlushError() error { return io.ErrClosedPipe }
func (w *failingFlushWriter) Flush()          { _ = w.FlushError() }
func (w *failingFlushWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return http.NewResponseController(w.ResponseWriter).Hijack()
}

func newTestForwardingHandler(t *testing.T, prov provider.Provider, m *metrics.Metrics) *forwardingHandler {
	t.Helper()
	h, transport, err := newForwardingHandler(prov, codertestutil.NewFakeSink(t).Logger(), m, noop.NewTracerProvider().Tracer(t.Name()))
	require.NoError(t, err)
	require.NotNil(t, h.proxy)
	t.Cleanup(transport.CloseIdleConnections)
	return h
}

func recordedRequest(t *testing.T, body io.Reader) *http.Request {
	t.Helper()
	ctx := aibcontext.AsActor(t.Context(), t.Name(), "actor@example.test", recorder.Metadata{"team": "engineering"})
	req := httptest.NewRequest(http.MethodPost, "/openai/v1/chat/completions", body).WithContext(ctx)
	req.Pattern = "/openai/v1/chat/completions"
	req.Header.Set("Authorization", "Bearer user-secret-key")
	return req
}

func serveForwardingRequest(h *forwardingHandler, w http.ResponseWriter, r *http.Request, rec recorder.Recorder) {
	if r.Body != nil && r.Body != http.NoBody {
		r.Body = http.MaxBytesReader(w, r.Body, routing.MaxRequestBodyBytes)
		defer r.Body.Close()
	}
	tasks := utils.NewConcurrentGroup()
	defer func() { _ = tasks.Wait() }()
	h.serve(context.WithoutCancel(r.Context()), w, r, rec, tasks)
}

func TestRouterRecorderDispatch(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		path   string
		status int
	}{
		{path: "/openai/v1/responses", status: http.StatusOK},
		{path: "/openai/v1//responses", status: http.StatusTemporaryRedirect},
		{path: "/unknown", status: http.StatusNotFound},
		{path: "*", status: http.StatusBadRequest},
	} {
		t.Run(tc.path, func(t *testing.T) {
			t.Parallel()
			rec := &lifecycleRecorder{}
			h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"}), nil)
			h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				require.Equal(t, http.StatusOK, tc.status, "redirects and routing errors must not forward")
				require.Equal(t, "/openai/v1/responses", r.Pattern)
				require.Equal(t, "/responses", r.URL.Path)
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: http.NoBody}, nil
			})
			router := &Router{mux: http.NewServeMux()}
			router.mux.Handle("/openai/v1/responses", forwardingHandlerFunc(h.serve))
			req := recordedRequest(t, nil)
			req.URL.Path, req.RequestURI = tc.path, tc.path
			response := httptest.NewRecorder()
			tasks := utils.NewConcurrentGroup()
			router.Serve(t.Context(), response, req, rec, tasks)
			require.NoError(t, tasks.Wait())
			require.Equal(t, tc.status, response.Code)
			if tc.status == http.StatusOK {
				require.Len(t, rec.starts, 1)
				require.Len(t, rec.ends, 1)
			} else {
				require.Empty(t, rec.starts)
				require.Empty(t, rec.ends)
			}
		})
	}
}

func TestForwardingHandlerLifecycle(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"Success", "BytesWithEOF", "StartFailure", "EndFailure", "TransportFailure", "Upstream503", "ReadFailure", "WriteFailure", "FlushFailure", "Panic", "Cancellation", "PoolFailure", "CircuitOpen"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			rec := &lifecycleRecorder{onEnd: func(ctx context.Context) {
				require.NoError(t, ctx.Err(), "finalization must detach request cancellation")
				deadline, ok := ctx.Deadline()
				require.True(t, ok)
				require.Positive(t, time.Until(deadline))
				require.LessOrEqual(t, time.Until(deadline), recorder.DefaultAsyncTimeout)
			}}
			body := &trackedBody{Reader: strings.NewReader("chunk")}
			input := &trackedBody{Reader: strings.NewReader("not JSON")}
			req := recordedRequest(t, input)
			ctx, cancel := context.WithCancel(req.Context())
			defer cancel()
			req = req.WithContext(ctx)
			response := httptest.NewRecorder()
			var writer http.ResponseWriter = response
			status, wantStatus := http.StatusOK, http.StatusOK
			var transportErr error
			var wantPanic any
			switch name {
			case "BytesWithEOF":
				body.Reader = readerFunc(func(p []byte) (int, error) { return copy(p, "chunk"), io.EOF })
			case "StartFailure":
				rec.startErr = xerrors.New("start unavailable")
			case "EndFailure":
				rec.endErr = xerrors.New("end unavailable")
			case "TransportFailure":
				transportErr = io.ErrClosedPipe
				wantStatus = http.StatusBadGateway
			case "Upstream503":
				status = http.StatusServiceUnavailable
				wantStatus = status
			case "ReadFailure":
				body.Reader = io.MultiReader(body.Reader, readerFunc(func([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }))
				wantPanic = http.ErrAbortHandler
			case "WriteFailure":
				writer = &failingWriter{response}
				wantPanic = http.ErrAbortHandler
			case "FlushFailure":
				writer = &failingFlushWriter{response}
				wantPanic = http.ErrAbortHandler
			case "Panic":
				body.Reader = readerFunc(func([]byte) (int, error) { panic("copy panic") })
				wantPanic = "copy panic"
			case "Cancellation":
				body.Reader = io.MultiReader(body.Reader, readerFunc(func([]byte) (int, error) { cancel(); return 0, context.Canceled }))
				wantPanic = http.ErrAbortHandler
			case "PoolFailure":
				transportErr = &keypool.Error{Kind: keypool.ErrorKindRateLimited}
				wantStatus = http.StatusBadGateway
			case "CircuitOpen":
				transportErr = circuitbreaker.ErrCircuitOpen
				wantStatus = http.StatusBadGateway
			}
			m := metrics.NewMetrics(prometheus.NewRegistry())
			h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"}), m)
			calls := 0
			h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				require.Zero(t, input.eofs)
				payload, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				require.Equal(t, "not JSON", string(payload))
				require.NotNil(t, r.GetBody)
				replay, err := r.GetBody()
				require.NoError(t, err)
				payload, err = io.ReadAll(replay)
				require.NoError(t, err)
				require.NoError(t, replay.Close())
				require.Equal(t, "not JSON", string(payload))
				if transportErr != nil {
					return nil, transportErr
				}
				return &http.Response{StatusCode: status, Header: http.Header{}, Body: body}, nil
			})
			serve := func() { serveForwardingRequest(h, writer, req, rec) }
			if wantPanic != nil {
				require.PanicsWithValue(t, wantPanic, serve)
			} else {
				serve()
				require.Equal(t, wantStatus, response.Code)
			}
			require.Equal(t, 1, input.closes)
			require.Equal(t, 1, input.eofs)
			require.Len(t, rec.starts, 1)
			if name == "StartFailure" {
				require.Equal(t, 1, calls)
				require.Equal(t, "chunk", response.Body.String())
				require.Empty(t, rec.ends)
				return
			}
			require.Equal(t, 1, calls)
			require.Len(t, rec.ends, 1)
			require.Equal(t, rec.starts[0].ID, rec.ends[0].ID)
			require.False(t, rec.starts[0].StartedAt.IsZero())
			require.False(t, rec.ends[0].EndedAt.Before(rec.starts[0].StartedAt))
			require.Empty(t, rec.starts[0].Model)
			require.Equal(t, credential.KindBYOK, rec.starts[0].CredentialKind)
			require.Equal(t, utils.MaskSecret("user-secret-key"), rec.starts[0].CredentialHint)
			metricStatus := metrics.InterceptionCountStatusFailed
			if name == "Success" || name == "BytesWithEOF" || name == "EndFailure" {
				require.Empty(t, rec.ends[0].ErrorType)
				require.Empty(t, rec.ends[0].ErrorMessage)
				require.Equal(t, "chunk", response.Body.String())
				metricStatus = metrics.InterceptionCountStatusCompleted
			} else {
				require.NotEmpty(t, rec.ends[0].ErrorType)
				require.NotEmpty(t, rec.ends[0].ErrorMessage)
			}
			if name == "Upstream503" {
				require.Equal(t, recorder.ErrorTypeServerError, rec.ends[0].ErrorType)
			}
			if name == "FlushFailure" {
				require.Equal(t, io.ErrClosedPipe.Error(), rec.ends[0].ErrorMessage)
			}
			if transportErr == nil {
				require.Equal(t, 1, body.closes)
			}
			require.Equal(t, 1.0, promtest.ToFloat64(m.InterceptionCount.WithLabelValues("openai", "", metricStatus, "/v1/chat/completions", http.MethodPost, t.Name(), string(client.Unknown))))
			require.Zero(t, promtest.CollectAndCount(m.PassthroughCount))
		})
	}
}

func TestForwardingHandlerWireAndFailover(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"BYOKWithAPIDump", "Pool", "ExhaustedPool"} {
		pooled := name != "BYOKWithAPIDump"
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			var compressed bytes.Buffer
			gz := gzip.NewWriter(&compressed)
			_, err := gz.Write([]byte("raw response"))
			require.NoError(t, err)
			require.NoError(t, gz.Close())
			keys := make(chan string, 2)
			var attempts atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				keys <- r.Header.Get("Authorization")
				attempt := attempts.Add(1)
				payload, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				assert.Equal(t, "  not JSON\x00\n", string(payload))
				assert.EqualValues(t, -1, r.ContentLength)
				assert.Equal(t, "/base/models/a%2Fb", r.URL.EscapedPath())
				assert.Equal(t, "configured=1&raw=a;b", r.URL.RawQuery)
				if pooled {
					assert.Equal(t, "claude-code/1.0", r.UserAgent())
					assert.Equal(t, "gzip", r.Header.Get("Accept-Encoding"))
				} else {
					assert.Empty(t, r.UserAgent())
					assert.Empty(t, r.Header.Get("Accept-Encoding"))
				}
				if pooled && (attempt == 1 || name == "ExhaustedPool") {
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				w.Header().Set("Content-Encoding", "gzip")
				_, _ = w.Write(compressed.Bytes())
			}))
			t.Cleanup(upstream.Close)
			var pool *keypool.Pool
			if pooled {
				pool, err = keypool.New("openai", []string{"first-secret-key", "second-secret-key"}, quartz.NewMock(t), nil)
				require.NoError(t, err)
			}
			dumpDir := ""
			if !pooled {
				dumpDir = t.TempDir()
			}
			h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: upstream.URL + "/base?configured=1", KeyPool: pool, APIDumpDir: dumpDir}), nil)
			rec := &lifecycleRecorder{}
			body := &trackedBody{Reader: strings.NewReader("  not JSON\x00\n")}
			req := recordedRequest(t, body)
			req.URL.Path = "/openai/v1/models/a/b"
			req.URL.RawPath = "/openai/v1/models/a%2Fb"
			req.URL.RawQuery = "raw=a;b"
			if pooled {
				req.Header.Del("Authorization")
				req.Header.Set("User-Agent", "claude-code/1.0")
				req.Header.Set("Accept-Encoding", "gzip")
			}
			req.Header.Set("X-Claude-Code-Session-Id", "session-one")
			req.Header.Set("X-Coder-Agent-Firewall-Session-Id", "e5f6a7b8-1234-5678-9abc-def012345678")
			req.Header.Set("X-Coder-Agent-Firewall-Sequence-Number", "42")
			original := req.Header.Clone()
			response := httptest.NewRecorder()
			serveForwardingRequest(h, response, req, rec)
			if name == "ExhaustedPool" {
				require.Equal(t, http.StatusBadGateway, response.Code)
			} else {
				require.Equal(t, http.StatusOK, response.Code)
				require.Equal(t, compressed.Bytes(), response.Body.Bytes())
				require.Equal(t, "gzip", response.Header().Get("Content-Encoding"))
			}
			require.Equal(t, original, req.Header)
			require.Equal(t, 1, body.eofs)
			require.Equal(t, 1, body.closes)
			require.Len(t, rec.starts, 1)
			require.Len(t, rec.ends, 1)
			if name == "ExhaustedPool" {
				require.Equal(t, recorder.ErrorTypeUnauthorized, rec.ends[0].ErrorType)
				require.Contains(t, rec.ends[0].ErrorMessage, "key")
			} else {
				require.Empty(t, rec.ends[0].ErrorType)
			}
			if dumpDir != "" {
				for _, suffix := range []string{apidump.SuffixRequest, apidump.SuffixResponse} {
					files, err := filepath.Glob(filepath.Join(dumpDir, "openai", "passthrough", "*"+suffix))
					require.NoError(t, err)
					require.Len(t, files, 1)
					data, err := os.ReadFile(files[0])
					require.NoError(t, err)
					require.NotContains(t, string(data), "user-secret-key")
					if suffix == apidump.SuffixRequest {
						require.Contains(t, string(data), "Authorization: "+utils.MaskSecret("Bearer user-secret-key"))
						require.Contains(t, string(data), "  not JSON\x00\n")
					} else {
						require.Contains(t, string(data), "HTTP/1.1 200 OK")
					}
				}
			}
			start := rec.starts[0]
			require.Equal(t, t.Name(), start.InitiatorID)
			require.Equal(t, recorder.Metadata{"team": "engineering"}, start.Metadata)
			require.Equal(t, new("e5f6a7b8-1234-5678-9abc-def012345678"), start.AgentFirewallSessionID)
			require.Equal(t, new(int32(42)), start.AgentFirewallSequenceNumber)
			require.Empty(t, start.Model)
			if pooled {
				require.Equal(t, "Bearer first-secret-key", codertestutil.RequireReceive(codertestutil.Context(t, codertestutil.WaitLong), t, keys))
				require.Equal(t, "Bearer second-secret-key", codertestutil.RequireReceive(codertestutil.Context(t, codertestutil.WaitLong), t, keys))
				require.Equal(t, credential.KindCentralized, start.CredentialKind)
				require.Equal(t, credential.HintFailoverKey, start.CredentialHint)
				require.Equal(t, utils.MaskSecret("second-secret-key"), rec.ends[0].CredentialHint)
				require.Equal(t, new("session-one"), start.ClientSessionID)
				require.Equal(t, string(client.ClaudeCode), start.Client)
			} else {
				require.Equal(t, "Bearer user-secret-key", codertestutil.RequireReceive(codertestutil.Context(t, codertestutil.WaitLong), t, keys))
			}
		})
	}
}

func TestForwardingHandlerStreamsAndCancels(t *testing.T) {
	t.Parallel()
	canceled, finalized := make(chan struct{}), make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: first\n\n")
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-r.Context().Done():
			close(canceled)
		case <-release:
		}
	}))
	t.Cleanup(upstream.Close)
	h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: upstream.URL}), nil)
	startedRecord := make(chan struct{})
	rec := &lifecycleRecorder{onStart: func(context.Context) { close(startedRecord) }, onEnd: func(context.Context) { close(finalized) }}
	tasks := utils.NewConcurrentGroup()
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := aibcontext.AsActor(r.Context(), t.Name(), "", nil)
		h.serve(t.Context(), w, r.WithContext(ctx), rec, tasks)
	}))
	t.Cleanup(gateway.Close)
	ctx, cancel := context.WithCancel(codertestutil.Context(t, codertestutil.WaitLong))
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, gateway.URL+"/openai/v1/chat/completions", strings.NewReader("request"))
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer user-secret-key")
	resp, err := gateway.Client().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	first := make([]byte, len("data: first\n\n"))
	_, err = io.ReadFull(resp.Body, first)
	require.NoError(t, err)
	require.Equal(t, "data: first\n\n", string(first))
	codertestutil.TryReceive(ctx, t, startedRecord)
	select {
	case <-finalized:
		t.Fatal("end recording ran before response delivery completed")
	default:
	}
	cancel()
	waitCtx := codertestutil.Context(t, codertestutil.WaitLong)
	codertestutil.TryReceive(waitCtx, t, canceled)
	codertestutil.TryReceive(waitCtx, t, finalized)
	require.NoError(t, tasks.Wait())
	require.Len(t, rec.ends, 1)
	require.NotEmpty(t, rec.ends[0].ErrorType)
}

func TestForwardingHandlerAsyncRecording(t *testing.T) {
	t.Parallel()
	for _, stage := range []string{"Start", "End"} {
		t.Run(stage, func(t *testing.T) {
			t.Parallel()
			ctx := codertestutil.Context(t, codertestutil.WaitShort)
			recording, release := make(chan struct{}), make(chan struct{})
			tasks := utils.NewConcurrentGroup()
			rec := &lifecycleRecorder{}
			defer func() {
				close(release)
				require.NoError(t, tasks.Wait())
				require.Len(t, rec.starts, 1)
				require.Len(t, rec.ends, 1)
				require.False(t, rec.ends[0].EndedAt.Before(rec.starts[0].StartedAt))
			}()
			block := func(ctx context.Context) {
				close(recording)
				select {
				case <-release:
				case <-ctx.Done():
					t.Error("recording timed out before the HTTP handler returned")
				}
			}
			if stage == "Start" {
				rec.onStart = block
			} else {
				rec.onEnd = block
			}
			h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"}), nil)
			h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				_, err := io.Copy(io.Discard, r.Body)
				if err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("response"))}, nil
			})
			response := httptest.NewRecorder()
			req := recordedRequest(t, strings.NewReader("request"))
			served := make(chan struct{})
			go func() {
				defer close(served)
				h.serve(ctx, response, req, rec, tasks)
			}()
			codertestutil.TryReceive(ctx, t, recording)
			codertestutil.TryReceive(ctx, t, served)
			require.Equal(t, http.StatusOK, response.Code)
			require.Equal(t, "response", response.Body.String())
		})
	}
}

func TestForwardingHandlerAbandonsCapture(t *testing.T) {
	t.Parallel()
	ctx := codertestutil.Context(t, codertestutil.WaitShort)
	input, output := io.Pipe()
	defer input.Close()
	defer output.Close()
	h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"}), nil)
	h.transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusBadRequest, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("upstream error"))}, nil
	})
	req := recordedRequest(t, input)
	req.Header.Set("User-Agent", "claude-code/1.0")
	rec := &lifecycleRecorder{}
	tasks := utils.NewConcurrentGroup()
	response := httptest.NewRecorder()
	done := make(chan error, 1)
	go func() {
		h.serve(ctx, response, req, rec, tasks)
		done <- tasks.Wait()
	}()
	require.NoError(t, codertestutil.RequireReceive(ctx, t, done))
	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Equal(t, "upstream error", response.Body.String())
	require.Len(t, rec.starts, 1)
	require.Nil(t, rec.starts[0].ClientSessionID)
	require.Len(t, rec.ends, 1)
	require.Equal(t, recorder.ErrorTypeBadRequest, rec.ends[0].ErrorType)
}

func TestForwardingHandlerProcessingPanic(t *testing.T) {
	t.Parallel()
	for _, stage := range []string{"Start", "End"} {
		t.Run(stage, func(t *testing.T) {
			t.Parallel()
			rec := &lifecycleRecorder{}
			panicRecord := func(context.Context) { panic("recorder panic") }
			if stage == "Start" {
				rec.onStart = panicRecord
			} else {
				rec.onEnd = panicRecord
			}
			h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"}), nil)
			h.transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("response"))}, nil
			})
			tasks := utils.NewConcurrentGroup()
			response := httptest.NewRecorder()
			h.serve(t.Context(), response, recordedRequest(t, nil), rec, tasks)
			require.ErrorContains(t, tasks.Wait(), "processing panicked")
			require.Equal(t, http.StatusOK, response.Code)
			require.Equal(t, "response", response.Body.String())
			require.Len(t, rec.starts, 1)
			if stage == "Start" {
				require.Empty(t, rec.ends)
			} else {
				require.Len(t, rec.ends, 1)
			}
		})
	}
}

func TestForwardingHandlerTee(t *testing.T) {
	t.Parallel()
	ctx := codertestutil.Context(t, codertestutil.WaitLong)
	input, output := io.Pipe()
	defer input.Close()
	defer output.Close()
	const prefix = `{"metadata":{"user_id":"user_hash_account_id_session_`
	const suffix = `session-one"}}`
	pool, err := keypool.New("openai", []string{"first-key", "second-key"}, quartz.NewMock(t), nil)
	require.NoError(t, err)
	h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test", KeyPool: pool}), nil)
	rec := &lifecycleRecorder{}
	readPrefix := make(chan time.Time, 1)
	attempts := 0
	h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		defer r.Body.Close()
		attempts++
		if attempts == 1 {
			chunk := make([]byte, len(prefix))
			_, err := io.ReadFull(r.Body, chunk)
			assert.NoError(t, err)
			assert.Equal(t, prefix, string(chunk))
			readPrefix <- time.Now()
			return &http.Response{StatusCode: http.StatusUnauthorized, Header: http.Header{}, Body: http.NoBody}, nil
		}
		payload, err := io.ReadAll(r.Body)
		if !assert.NoError(t, err) {
			return nil, err
		}
		assert.Equal(t, prefix+suffix, string(payload), "the retry must replay the prefix and read the remaining input")
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: http.NoBody}, nil
	})
	req := recordedRequest(t, input)
	req.Header.Del("Authorization")
	req.Header.Set("User-Agent", "claude-code/1.0")
	response := httptest.NewRecorder()
	served := make(chan struct{})
	before := time.Now()
	go func() {
		defer close(served)
		serveForwardingRequest(h, response, req, rec)
	}()
	_, err = io.WriteString(output, prefix)
	require.NoError(t, err)
	forwardedAt := codertestutil.RequireReceive(ctx, t, readPrefix)
	_, err = io.WriteString(output, suffix)
	require.NoError(t, err)
	require.NoError(t, output.Close())
	codertestutil.TryReceive(ctx, t, served)
	require.Equal(t, http.StatusOK, response.Code)
	require.Equal(t, 2, attempts)
	require.Len(t, rec.starts, 1)
	require.Len(t, rec.ends, 1)
	require.Equal(t, new("session-one"), rec.starts[0].ClientSessionID)
	require.False(t, rec.starts[0].StartedAt.Before(before))
	require.False(t, rec.starts[0].StartedAt.After(forwardedAt))
	require.Empty(t, rec.ends[0].ErrorType)
}

func TestRequestBufferBytes(t *testing.T) {
	t.Parallel()
	t.Run("Empty", func(t *testing.T) {
		t.Parallel()
		b := &requestBuffer{source: http.NoBody, err: io.EOF}
		reader, err := b.getBody()
		require.NoError(t, err)
		require.Equal(t, http.NoBody, reader, "known empty bodies must not become chunked requests")
		payload, err := b.bytes()
		require.NoError(t, err)
		require.Empty(t, payload)
	})
	input := &trackedBody{Reader: strings.NewReader("payload")}
	b := &requestBuffer{source: input}
	reader, err := b.getBody()
	require.NoError(t, err)
	t.Cleanup(func() { _ = reader.Close() })

	for range 2 {
		payload, err := b.bytes()
		require.NoError(t, err)
		require.Equal(t, "payload", string(payload))
	}
	payload, err := io.ReadAll(reader)
	require.NoError(t, err)
	require.Equal(t, "payload", string(payload), "filling the buffer must not advance another reader")
	require.Equal(t, 1, input.eofs, "completed input must not be read again")
}

func TestRequestBufferReaders(t *testing.T) {
	t.Parallel()
	for _, terminal := range []error{io.EOF, io.ErrClosedPipe} {
		t.Run(terminal.Error(), func(t *testing.T) {
			t.Parallel()
			ctx := codertestutil.Context(t, codertestutil.WaitLong)
			reading, release := make(chan struct{}), make(chan struct{})
			defer close(release)
			calls := 0
			b := &requestBuffer{captured: make(chan capturedBody, 1), source: readerFunc(func(p []byte) (int, error) {
				calls++
				if calls == 1 {
					return copy(p, "prefix"), nil
				}
				close(reading)
				<-release
				return copy(p, "suffix"), terminal
			})}
			first, err := b.getBody()
			require.NoError(t, err)
			defer first.Close()
			prefix := make([]byte, len("prefix"))
			_, err = io.ReadFull(first, prefix)
			require.NoError(t, err)
			done := make(chan struct{}, 2)
			go func() {
				payload, err := b.bytes()
				if errors.Is(terminal, io.EOF) {
					assert.NoError(t, err)
					assert.Equal(t, "prefixsuffix", string(payload))
				} else {
					assert.ErrorIs(t, err, terminal)
					assert.Nil(t, payload)
				}
				done <- struct{}{}
			}()
			codertestutil.TryReceive(ctx, t, reading)
			b.finishCapture()
			capture := codertestutil.RequireReceive(ctx, t, b.captured)
			require.ErrorIs(t, capture.err, io.ErrUnexpectedEOF, "abandoning observation must not wait for a source read")
			require.Nil(t, capture.payload)
			second, err := b.getBody()
			require.NoError(t, err)
			defer second.Close()
			go func() {
				_, err := io.ReadFull(second, prefix)
				assert.NoError(t, err)
				assert.Equal(t, "prefix", string(prefix))
				done <- struct{}{}
			}()
			// A cached prefix must remain readable while another reader is
			// blocked waiting for input at the buffered end.
			codertestutil.RequireReceive(ctx, t, done)
			release <- struct{}{}
			payload, err := io.ReadAll(second)
			require.Equal(t, "suffix", string(payload))
			if errors.Is(terminal, io.EOF) {
				require.NoError(t, err)
				payload, err = b.bytes()
				require.NoError(t, err)
				require.Equal(t, "prefixsuffix", string(payload))
			} else {
				require.ErrorIs(t, err, terminal)
				_, err = b.bytes()
				require.ErrorIs(t, err, terminal)
			}
			codertestutil.RequireReceive(ctx, t, done)
			require.Equal(t, 2, calls, "input bytes must be captured only once")
		})
	}
}

func TestForwardingHandlerProviderCredentials(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"OpenAIConnectionAuth", "AnthropicAPIKey", "CopilotBearer"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			pool, err := keypool.New("test", []string{"pool-secret-key"}, quartz.NewMock(t), nil)
			require.NoError(t, err)
			var prov provider.Provider = provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test", KeyPool: pool})
			rec := &lifecycleRecorder{}
			req := recordedRequest(t, strings.NewReader("body"))
			wantAuth := http.Header{"Authorization": {"Bearer user-secret-key"}}
			switch name {
			case "OpenAIConnectionAuth":
				req.Header.Set("Connection", "Authorization")
			case "AnthropicAPIKey":
				prov, err = provider.NewAnthropic(t.Context(), config.Anthropic{BaseURL: "https://upstream.example.test", KeyPool: pool}, nil, nil)
				require.NoError(t, err)
				req.Header.Set("X-Api-Key", "anthropic-secret-key")
				wantAuth = http.Header{"X-Api-Key": {"anthropic-secret-key"}}
			case "CopilotBearer":
				prov = provider.NewCopilot(config.Copilot{BaseURL: "https://upstream.example.test"})
			}
			h := newTestForwardingHandler(t, prov, nil)
			calls := 0
			h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				for _, header := range []string{"Authorization", "X-Api-Key"} {
					require.Equal(t, wantAuth.Values(header), r.Header.Values(header))
				}
				return &http.Response{StatusCode: http.StatusUnauthorized, Header: http.Header{}, Body: http.NoBody}, nil
			})
			response := httptest.NewRecorder()
			serveForwardingRequest(h, response, req, rec)
			require.Equal(t, http.StatusUnauthorized, response.Code)
			require.Equal(t, 1, calls, "BYOK failures must not fall back to pooled credentials")
			require.Len(t, rec.starts, 1)
			require.Len(t, rec.ends, 1)
			require.Equal(t, credential.KindBYOK, rec.starts[0].CredentialKind)
			require.Equal(t, recorder.ErrorTypeUnauthorized, rec.ends[0].ErrorType)
		})
	}
}

func TestForwardingHandlerCircuitBreaker(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusEarlyHints)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(upstream.Close)
	cfg := config.DefaultCircuitBreaker()
	cfg.FailureThreshold = 1
	cfg.Timeout = time.Hour
	h := newTestForwardingHandler(t, provider.NewOpenAI(config.OpenAI{BaseURL: upstream.URL, CircuitBreaker: &cfg}), nil)
	for range 2 {
		rec := &lifecycleRecorder{}
		response := httptest.NewRecorder()
		serveForwardingRequest(h, response, recordedRequest(t, nil), rec)
		require.Len(t, rec.starts, 1)
		require.Len(t, rec.ends, 1)
		require.Equal(t, recorder.ErrorTypeServerError, rec.ends[0].ErrorType)
	}
	require.EqualValues(t, 1, calls.Load(), "the final 503 must trip the breaker despite informational responses")
}

func TestForwardingHandlerRejectsRequest(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"MissingActor", "MissingRecorder", "WebSocket", "Firewall", "DeclaredOversize", "UnknownLengthOversize", "BodyReadFailure", "UnsupportedSigning"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			rec := &lifecycleRecorder{}
			var requestRecorder recorder.Recorder = rec
			req := recordedRequest(t, strings.NewReader("body"))
			var prov provider.Provider = provider.NewOpenAI(config.OpenAI{BaseURL: "https://upstream.example.test"})
			status := http.StatusBadRequest
			switch name {
			case "MissingActor":
				req = req.WithContext(t.Context())
			case "MissingRecorder":
				requestRecorder = nil
				status = http.StatusInternalServerError
			case "WebSocket":
				req.Method = http.MethodGet
				req.Header.Set("Connection", "Upgrade")
				req.Header.Set("Upgrade", "websocket")
				status = http.StatusNotImplemented
			case "Firewall":
				req.Header.Set("X-Coder-Agent-Firewall-Session-Id", "invalid")
			case "DeclaredOversize":
				req.ContentLength = routing.MaxRequestBodyBytes + 1
				status = http.StatusRequestEntityTooLarge
			case "UnknownLengthOversize":
				req.Body = io.NopCloser(io.LimitReader(readerFunc(func(p []byte) (int, error) { clear(p); return len(p), nil }), routing.MaxRequestBodyBytes+1))
				req.ContentLength = -1
				status = http.StatusRequestEntityTooLarge
			case "BodyReadFailure":
				req.Body = io.NopCloser(readerFunc(func([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }))
				req.ContentLength = -1
				status = http.StatusBadGateway
			case "UnsupportedSigning":
				var err error
				prov, err = provider.NewAnthropic(t.Context(), config.Anthropic{BaseURL: "https://upstream.example.test"}, nil, &config.AWSClaudePlatform{Region: "us-west-2", WorkspaceID: "wrkspc_test"})
				require.NoError(t, err)
				req.Header.Del("Authorization")
				status = http.StatusNotImplemented
			}
			h := newTestForwardingHandler(t, prov, nil)
			bodyFailure := name == "UnknownLengthOversize" || name == "BodyReadFailure"
			h.transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				require.True(t, bodyFailure, "unexpected upstream request")
				_, err := io.Copy(io.Discard, r.Body)
				return nil, err
			})
			response := httptest.NewRecorder()
			serveForwardingRequest(h, response, req, requestRecorder)
			require.Equal(t, status, response.Code)
			if bodyFailure {
				require.Len(t, rec.starts, 1)
				require.Len(t, rec.ends, 1)
				require.NotEmpty(t, rec.starts[0].ID)
				require.Nil(t, rec.starts[0].ClientSessionID)
				require.NotEmpty(t, rec.ends[0].ErrorType)
			} else {
				require.Empty(t, rec.starts)
				require.Empty(t, rec.ends)
			}
		})
	}
}
