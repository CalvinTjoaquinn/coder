package aibridged

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/xerrors"
	"storj.io/drpc"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/coder/v2/aibridge"
	"github.com/coder/coder/v2/aibridge/recorder"
	"github.com/coder/coder/v2/aibridge/utils"
	"github.com/coder/coder/v2/codersdk"
	"github.com/coder/coder/v2/testutil"
)

// blockingHandler returns a handler that signals started, then blocks until
// release is closed or the request context is canceled.
func blockingHandler() (h http.Handler, started <-chan struct{}, release func()) {
	startedCh := make(chan struct{}, 1)
	releaseCh := make(chan struct{})
	h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startedCh <- struct{}{}
		select {
		case <-releaseCh:
			w.WriteHeader(http.StatusNoContent)
		case <-r.Context().Done():
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	return h, startedCh, func() { close(releaseCh) }
}

// serveAsync invokes an acquired handler and reports the response status.
func serveAsync(handler http.Handler) <-chan int {
	done := make(chan int, 1)
	go func() {
		defer close(done)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", nil))
		done <- rec.Code
	}()
	return done
}

type proxyTestConn struct {
	drpc.Conn
	closed chan struct{}
	once   sync.Once
}

func (c *proxyTestConn) Closed() <-chan struct{} { return c.closed }
func (c *proxyTestConn) Close() error {
	c.once.Do(func() { close(c.closed) })
	return nil
}

// DRPC stays available, including reconnects, until requests finish or shutdown
// is canceled. An in-progress reload must not block reconnecting or closing DRPC.
func TestServerProxy_Shutdown(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name       string
		cancel     bool
		wantStatus int
		wantErr    error
	}{
		{name: "Graceful", wantStatus: http.StatusNoContent},
		{name: "Canceled", cancel: true, wantStatus: http.StatusServiceUnavailable, wantErr: context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			upstreamHandler, started, release := blockingHandler()
			tracker := newProxyTestServer(t, upstreamHandler)
			release = sync.OnceFunc(release)
			t.Cleanup(release)
			testCtx := testutil.Context(t, testutil.WaitShort)
			first := &Client{Conn: &proxyTestConn{closed: make(chan struct{})}}
			second := &Client{Conn: &proxyTestConn{closed: make(chan struct{})}}
			reconnected := make(chan struct{})
			dials := 0
			tracker.clientCh = make(chan DRPCClient)
			tracker.clientDialer = func(context.Context) (DRPCClient, error) {
				dials++
				switch dials {
				case 1:
					return first, nil
				case 2:
					return nil, xerrors.New("transient dial failure")
				default:
					close(reconnected)
					return second, nil
				}
			}
			tracker.wg.Add(1)
			go tracker.connect()
			client, err := tracker.Client(testCtx)
			require.NoError(t, err)
			require.Same(t, first, client)

			handler, err := tracker.GetRequestHandler(testCtx, Request{})
			require.NoError(t, err)
			inflightReqDone := serveAsync(handler)
			testutil.RequireReceive(testCtx, t, started)

			shutdownCtx, cancel := context.WithCancel(testCtx)
			defer cancel()
			shutdownDone := make(chan error, 1)
			go func() { shutdownDone <- tracker.Shutdown(shutdownCtx) }()
			require.Eventually(t, tracker.shuttingDown.Load, testutil.WaitShort, testutil.IntervalFast)

			refused, err := tracker.GetRequestHandler(testCtx, Request{})
			require.NoError(t, err)
			rec := httptest.NewRecorder()
			refused.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", nil))
			require.Equal(t, http.StatusServiceUnavailable, rec.Code)
			require.Equal(t, "AI Gateway is shutting down\n", rec.Body.String())
			select {
			case err := <-shutdownDone:
				t.Fatalf("shutdown returned before the request finished: %v", err)
			default:
			}

			client, err = tracker.Client(testCtx)
			require.NoError(t, err)
			require.Same(t, first, client, "DRPC must remain available during drainage")
			tracker.backendMu.Lock()
			defer tracker.backendMu.Unlock()
			require.NoError(t, first.DRPCConn().Close())
			testutil.TryReceive(testCtx, t, reconnected)
			client, err = tracker.Client(testCtx)
			require.NoError(t, err)
			require.Same(t, second, client, "DRPC must reconnect during drainage")
			require.NoError(t, tracker.lifecycleCtx.Err())

			if tc.cancel {
				cancel()
			} else {
				release()
			}
			require.Equal(t, tc.wantStatus, testutil.RequireReceive(testCtx, t, inflightReqDone))
			require.ErrorIs(t, testutil.RequireReceive(testCtx, t, shutdownDone), tc.wantErr)
			require.ErrorIs(t, context.Cause(tracker.lifecycleCtx), ErrShutdown)
			testutil.TryReceive(testCtx, t, second.DRPCConn().Closed())
			require.False(t, tracker.Ready())
		})
	}
}

func TestServerProxy_CancelsWithLifecycle(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	tracker, err := New(ctx, func(ctx context.Context) (DRPCClient, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}, slogtest.Make(t, nil), nil, codersdk.Experiments{codersdk.ExperimentAIGatewayReverseProxy}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = tracker.Close() })
	handler, started, release := blockingHandler()
	t.Cleanup(release)
	tracker.backend.Store(&backend{proxyRouter: func(recorder.Recorder) http.Handler {
		return tracker.inflight.Middleware(handler)
	}})
	acquired, err := tracker.GetRequestHandler(t.Context(), Request{})
	require.NoError(t, err)
	done := serveAsync(acquired)
	testCtx := testutil.Context(t, testutil.WaitShort)
	testutil.RequireReceive(testCtx, t, started)

	cancel()
	require.Equal(t, http.StatusServiceUnavailable, testutil.RequireReceive(testCtx, t, done))
	require.ErrorIs(t, tracker.Err(), context.Canceled)
	refused, err := tracker.GetRequestHandler(testCtx, Request{})
	require.NoError(t, err)
	for _, h := range []http.Handler{acquired, refused} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", nil))
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)
		require.Equal(t, "AI Gateway is shutting down\n", rec.Body.String())
	}
}

// TestServerProxy_DeadlineDoesNotWaitForHandler ensures shutdown returns
// even if an admitted handler cannot observe cancellation, such as a blocked
// response write.
func TestServerProxy_DeadlineDoesNotWaitForHandler(t *testing.T) {
	t.Parallel()

	started := make(chan context.Context, 1)
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	tracker := newProxyTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started <- r.Context()
		<-release
		w.WriteHeader(http.StatusNoContent)
	}))
	handler, err := tracker.GetRequestHandler(t.Context(), Request{})
	require.NoError(t, err)
	inflightReqDone := serveAsync(handler)
	t.Cleanup(func() {
		unblock()
		select {
		case <-inflightReqDone:
		case <-time.After(testutil.WaitLong):
			t.Error("handler did not finish after release")
		}
	})
	testCtx := testutil.Context(t, testutil.WaitLong)
	requestCtx := testutil.RequireReceive(testCtx, t, started)

	ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(-time.Second))
	defer cancel()
	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- tracker.Shutdown(ctx) }()
	require.ErrorIs(t, testutil.RequireReceive(testCtx, t, shutdownDone), context.DeadlineExceeded)
	testutil.TryReceive(testCtx, t, requestCtx.Done())
	require.ErrorIs(t, requestCtx.Err(), context.Canceled)

	select {
	case <-inflightReqDone:
		t.Fatal("handler returned before release")
	default:
	}

	// Release the test handler so incorrect re-admission fails rather than hangs.
	unblock()
	require.Equal(t, http.StatusNoContent, testutil.RequireReceive(testCtx, t, inflightReqDone))
	refused, err := tracker.GetRequestHandler(testCtx, Request{})
	require.NoError(t, err)
	for _, h := range []http.Handler{handler, refused} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", nil))
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)
		require.Equal(t, "AI Gateway is shutting down\n", rec.Body.String())
	}
}

// TestServerProxy_RefusesAfterShutdown asserts no request is admitted once
// shutdown has begun.
func TestServerProxy_RefusesAfterShutdown(t *testing.T) {
	t.Parallel()

	tracker := newProxyTestServer(t, http.NotFoundHandler())
	acquired, err := tracker.GetRequestHandler(t.Context(), Request{})
	require.NoError(t, err)
	require.NoError(t, tracker.Shutdown(t.Context()))
	refused, err := tracker.GetRequestHandler(t.Context(), Request{})
	require.NoError(t, err)

	for _, tc := range []struct {
		name    string
		handler http.Handler
	}{
		{name: "Retained", handler: acquired},
		{name: "AcquiredAfterShutdown", handler: refused},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := httptest.NewRecorder()
			tc.handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", nil))
			require.Equal(t, http.StatusServiceUnavailable, rec.Code)
			require.Equal(t, "AI Gateway is shutting down\n", rec.Body.String())
		})
	}
}

// newProxyTestServer publishes a test handler behind the proxy admission gate.
// It registers cleanup but leaves DRPC setup and the connection loop to the test.
func newProxyTestServer(t *testing.T, handler http.Handler) *Server {
	t.Helper()
	ctx, cancel := context.WithCancelCause(t.Context())
	s := &Server{
		lifecycleCtx: ctx,
		cancelFn:     cancel,
		logger:       slogtest.Make(t, nil),
		inflight:     aibridge.NewInflightGate(slogtest.Make(t, nil)),
	}
	s.backend.Store(&backend{proxyRouter: func(recorder.Recorder) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			s.serveProxy(w, r, func(_ context.Context, w http.ResponseWriter, r *http.Request, _ *utils.ConcurrentGroup) {
				handler.ServeHTTP(w, r)
			})
		})
	}})
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// TestServerProxy_ProcessingOutlivesHandler asserts processing tasks do not
// delay the response, keep request values, and hold shutdown drainage while
// DRPC stays live. Forced shutdown cancels their context.
func TestServerProxy_ProcessingOutlivesHandler(t *testing.T) {
	t.Parallel()

	type ctxKey struct{}
	for _, tc := range []struct {
		name   string
		panics bool
		forced bool
	}{
		{name: "Graceful"},
		{name: "Panic", panics: true},
		{name: "Forced", forced: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			testCtx := testutil.Context(t, testutil.WaitShort)
			taskStarted := make(chan context.Context, 1)
			releaseTask := make(chan struct{})
			release := sync.OnceFunc(func() { close(releaseTask) })
			t.Cleanup(release)
			tracker := newProxyTestServer(t, nil)
			sink := testutil.NewFakeSink(t)
			tracker.logger = sink.Logger()
			tracker.backend.Store(&backend{proxyRouter: func(recorder.Recorder) http.Handler {
				return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					tracker.serveProxy(w, r, func(processCtx context.Context, w http.ResponseWriter, _ *http.Request, tasks *utils.ConcurrentGroup) {
						tasks.Go(func() error {
							taskStarted <- processCtx
							select {
							case <-releaseTask:
								return nil
							case <-processCtx.Done():
								return processCtx.Err()
							}
						})
						if tc.panics {
							panic(http.ErrAbortHandler)
						}
						w.WriteHeader(http.StatusNoContent)
					})
				})
			}})
			handler, err := tracker.GetRequestHandler(testCtx, Request{})
			require.NoError(t, err)

			// The handler returns while its task is still blocked.
			reqCtx, cancelReq := context.WithCancel(context.WithValue(testCtx, ctxKey{}, "attribution"))
			rec := httptest.NewRecorder()
			served := make(chan any, 1)
			go func() {
				defer func() { served <- recover() }()
				req := httptest.NewRequest(http.MethodPost, "/", nil).WithContext(reqCtx)
				req.Header.Set("Authorization", "Bearer request-secret")
				handler.ServeHTTP(rec, req)
			}()
			if panicValue := testutil.RequireReceive(testCtx, t, served); tc.panics {
				require.Equal(t, http.ErrAbortHandler, panicValue)
			} else {
				require.Nil(t, panicValue)
				require.Equal(t, http.StatusNoContent, rec.Code)
			}
			cancelReq()
			processCtx := testutil.RequireReceive(testCtx, t, taskStarted)
			require.Equal(t, "attribution", processCtx.Value(ctxKey{}))
			require.NoError(t, processCtx.Err(), "request cancellation must not cancel processing")

			shutdownCtx, cancelShutdown := context.WithCancel(testCtx)
			defer cancelShutdown()
			shutdownDone := make(chan error, 1)
			go func() { shutdownDone <- tracker.Shutdown(shutdownCtx) }()
			require.Eventually(t, tracker.shuttingDown.Load, testutil.WaitShort, testutil.IntervalFast)
			select {
			case err := <-shutdownDone:
				t.Fatalf("shutdown returned before processing finished: %v", err)
			default:
			}
			require.NoError(t, tracker.lifecycleCtx.Err(), "DRPC must remain available while processing drains")

			if tc.forced {
				cancelShutdown()
				require.ErrorIs(t, testutil.RequireReceive(testCtx, t, shutdownDone), context.Canceled)
				testutil.TryReceive(testCtx, t, processCtx.Done())
				// The canceled task's error is logged with bounded request fields only.
				var entry slog.SinkEntry
				require.Eventually(t, func() bool {
					entries := sink.Entries(func(e slog.SinkEntry) bool { return e.Message == "proxy request processing failed" })
					if len(entries) != 1 {
						return false
					}
					entry = entries[0]
					return true
				}, testutil.WaitShort, testutil.IntervalFast)
				names := make([]string, 0, len(entry.Fields))
				for _, field := range entry.Fields {
					names = append(names, field.Name)
					require.NotContains(t, fmt.Sprint(field.Value), "request-secret")
				}
				require.ElementsMatch(t, []string{"method", "path", "error"}, names)
			} else {
				release()
				require.NoError(t, testutil.RequireReceive(testCtx, t, shutdownDone))
			}
			require.ErrorIs(t, context.Cause(tracker.lifecycleCtx), ErrShutdown)
		})
	}
}
