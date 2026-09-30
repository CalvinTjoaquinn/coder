package proxy_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace/noop"

	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/coder/v2/aibridge/config"
	aibcontext "github.com/coder/coder/v2/aibridge/context"
	"github.com/coder/coder/v2/aibridge/internal/testutil"
	"github.com/coder/coder/v2/aibridge/keypool"
	"github.com/coder/coder/v2/aibridge/metrics"
	"github.com/coder/coder/v2/aibridge/provider"
	"github.com/coder/coder/v2/aibridge/recorder"
	"github.com/coder/coder/v2/aibridge/routing"
	"github.com/coder/coder/v2/aibridge/utils"
	"github.com/coder/coder/v2/aibridge/x/proxy"
)

type keyPoolProvider struct {
	provider.Provider
	pool *keypool.Pool
}

func (p keyPoolProvider) KeyPool() *keypool.Pool { return p.pool }

func TestNewRouterValidatesProviders(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		providers   []provider.Provider
		errContains string
	}{
		{
			name:      "ValidNames",
			providers: []provider.Provider{&testutil.MockProvider{NameStr: "openai"}, &testutil.MockProvider{NameStr: "anthropic-eu-1"}},
		},
		{
			name:        "InvalidName",
			providers:   []provider.Provider{&testutil.MockProvider{NameStr: "OpenAI_1"}},
			errContains: `invalid provider name "OpenAI_1"`,
		},
		{
			name:        "DuplicateName",
			providers:   []provider.Provider{&testutil.MockProvider{NameStr: "openai"}, &testutil.MockProvider{NameStr: "openai"}},
			errContains: `duplicate provider name: "openai"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			router, err := proxy.NewRouter(tc.providers, slogtest.Make(t, nil), nil, noop.NewTracerProvider().Tracer(t.Name()))
			if tc.errContains != "" {
				require.ErrorContains(t, err, tc.errContains)
				require.Nil(t, router)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, router)
		})
	}
}

// Disabled providers serve the 503 sentinel independently of enabled routes.
func TestRouterDisabledProvider(t *testing.T) {
	t.Parallel()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "upstream")
	}))
	t.Cleanup(upstream.Close)
	enabled := &testutil.MockProvider{
		NameStr:     "openai",
		URL:         upstream.URL,
		Bridged:     []string{"/v1/chat/completions"},
		Passthrough: []string{"/v1/models"},
	}
	router, err := proxy.NewRouter(
		[]provider.Provider{enabled, provider.NewDisabledStub("disabled-openai", "openai")},
		slogtest.Make(t, nil), nil, noop.NewTracerProvider().Tracer(t.Name()),
	)
	require.NoError(t, err)

	for _, tc := range []struct {
		name       string
		path       string
		wantStatus int
		wantBody   string
	}{
		{name: "DisabledBridgedRoute", path: "/disabled-openai/v1/chat/completions", wantStatus: http.StatusServiceUnavailable, wantBody: routing.ErrorCodeProviderDisabled},
		{name: "DisabledPassthroughRoute", path: "/disabled-openai/v1/models", wantStatus: http.StatusServiceUnavailable, wantBody: routing.ErrorCodeProviderDisabled},
		{name: "DisabledUnknownRoute", path: "/disabled-openai/anything/else", wantStatus: http.StatusServiceUnavailable, wantBody: routing.ErrorCodeProviderDisabled},
		{name: "EnabledBridgedRoute", path: "/openai/v1/chat/completions", wantStatus: http.StatusBadRequest, wantBody: "actor"},
		{name: "EnabledPassthroughRoute", path: "/openai/v1/models", wantStatus: http.StatusOK, wantBody: "upstream"},
		{name: "UnknownEnabledRoute", path: "/openai/admin", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
		{name: "UnknownProvider", path: "/unknown/v1/models", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
		{name: "Root", path: "/", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp := httptest.NewRecorder()
			router.Serve(t.Context(), resp, httptest.NewRequest(http.MethodPost, tc.path, nil), nil, nil)

			assert.Equal(t, tc.wantStatus, resp.Code)
			assert.Contains(t, resp.Body.String(), tc.wantBody)
		})
	}
}

func TestRouterRegistersProviderRoutes(t *testing.T) {
	t.Parallel()

	var calls, connections atomic.Int32
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		assert.Contains(t, []string{"/base/models", "/base/tokens"}, r.URL.Path)
		assert.Equal(t, "configured=1&client=2", r.URL.RawQuery)
		assert.Equal(t, "192.0.2.1, 192.0.2.2", r.Header.Get("X-Forwarded-For"))
		assert.Equal(t, "gateway.example.test", r.Header.Get("X-Forwarded-Host"))
		assert.Equal(t, "http", r.Header.Get("X-Forwarded-Proto"))
		assert.Equal(t, "Bearer user-key", r.Header.Get("Authorization"))
		wantAgent := "aibridge"
		if r.URL.Path == "/base/tokens" {
			wantAgent = "client/1.0"
		}
		assert.Equal(t, wantAgent, r.Header.Get("User-Agent"))
		w.Header().Set("Set-Cookie", "custom=value")
		_, _ = io.WriteString(w, "upstream")
	}))
	upstream.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	upstream.Start()
	t.Cleanup(upstream.Close)
	prov := &testutil.MockProvider{
		NameStr: "test", URL: upstream.URL + "/base?configured=1",
		Bridged: []string{"/messages", "/models/bridged"}, Passthrough: []string{"/models", "/models/", "/tokens"},
	}
	m := metrics.NewMetrics(prometheus.NewRegistry())
	router, err := proxy.NewRouter([]provider.Provider{prov}, slogtest.Make(t, nil), m, noop.NewTracerProvider().Tracer(t.Name()))
	require.NoError(t, err)

	for _, path := range []string{"/messages", "/models/bridged", "/models", "/tokens"} {
		req := httptest.NewRequest(http.MethodGet, "http://gateway.example.test/test"+path+"?client=2", nil)
		req.RemoteAddr = "192.0.2.2:1234"
		req.Header.Set("Authorization", "Bearer user-key")
		req.Header.Set("X-Forwarded-For", "192.0.2.1")
		if path == "/tokens" {
			req.Header.Set("User-Agent", "client/1.0")
		}
		response := httptest.NewRecorder()
		router.Serve(t.Context(), response, req, nil, nil)
		if path == "/messages" || path == "/models/bridged" {
			require.Equal(t, http.StatusBadRequest, response.Code)
			require.Zero(t, calls.Load())
			continue
		}
		require.Equal(t, http.StatusOK, response.Code)
		require.Equal(t, "upstream", response.Body.String())
		require.Equal(t, "custom=value", response.Header().Get("Set-Cookie"))
	}
	require.EqualValues(t, 2, calls.Load())
	require.EqualValues(t, 1, connections.Load(), "passthrough routes must share the provider transport")
	require.NoError(t, promtestutil.CollectAndCompare(m.PassthroughCount, strings.NewReader(`
# HELP passthrough_total The count of requests which were not intercepted but passed through to the upstream.
# TYPE passthrough_total counter
passthrough_total{method="GET",provider="test",route="/models"} 1
passthrough_total{method="GET",provider="test",route="/tokens"} 1
`)))
}

// Only lifecycle methods are implemented; content recording must not occur.
type routerRecorder struct {
	recorder.Recorder
	starts []recorder.InterceptionRecord
	ends   []recorder.InterceptionRecordEnded
}

func (r *routerRecorder) RecordInterception(_ context.Context, record *recorder.InterceptionRecord) error {
	r.starts = append(r.starts, *record)
	return nil
}

func (r *routerRecorder) RecordInterceptionEnded(_ context.Context, record *recorder.InterceptionRecordEnded) error {
	r.ends = append(r.ends, *record)
	return nil
}

func TestRouterRecordedForwardingIsolation(t *testing.T) {
	t.Parallel()
	var connections atomic.Int32
	upstream := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Contains(t, []string{"/chat/completions", "/responses"}, r.URL.Path)
		_, _ = io.Copy(w, r.Body)
	}))
	upstream.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			connections.Add(1)
		}
	}
	upstream.Start()
	t.Cleanup(upstream.Close)
	m := metrics.NewMetrics(prometheus.NewRegistry())
	router, err := proxy.NewRouter([]provider.Provider{provider.NewOpenAI(config.OpenAI{BaseURL: upstream.URL})}, slogtest.Make(t, nil), m, noop.NewTracerProvider().Tracer(t.Name()))
	require.NoError(t, err)
	t.Cleanup(router.CloseIdleConnections)
	request := func(t *testing.T, route, actor string) {
		t.Helper()
		rec := &routerRecorder{}
		ctx := aibcontext.AsActor(t.Context(), actor, "", nil)
		req := httptest.NewRequest(http.MethodPost, "/openai/v1"+route, strings.NewReader(actor)).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer "+actor)
		response := httptest.NewRecorder()
		tasks := utils.NewConcurrentGroup()
		router.Serve(ctx, response, req, rec, tasks)
		require.NoError(t, tasks.Wait())
		require.Equal(t, http.StatusOK, response.Code)
		require.Equal(t, actor, response.Body.String())
		require.Len(t, rec.starts, 1)
		require.Len(t, rec.ends, 1)
		require.Equal(t, actor, rec.starts[0].InitiatorID)
		require.NotEmpty(t, rec.starts[0].ID)
		require.Equal(t, rec.starts[0].ID, rec.ends[0].ID)
		require.Empty(t, rec.starts[0].Model)
		require.Empty(t, rec.ends[0].ErrorType)
		require.Equal(t, 1.0, promtestutil.ToFloat64(m.InterceptionCount.WithLabelValues("openai", "", "completed", "/v1"+route, http.MethodPost, actor, "Unknown")))
	}
	request(t, "/chat/completions", "first-actor")
	request(t, "/responses", "second-actor")
	require.EqualValues(t, 1, connections.Load(), "bridged routes share one transport")
	router.CloseIdleConnections()
	request(t, "/responses", "after-close")
	require.EqualValues(t, 2, connections.Load(), "owned transport idle connections are closed")
	require.Zero(t, promtestutil.CollectAndCount(m.PassthroughCount))
	t.Run("Concurrent", func(t *testing.T) {
		t.Parallel()
		for i := range 8 {
			t.Run(fmt.Sprint(i), func(t *testing.T) {
				t.Parallel()
				request(t, "/chat/completions", t.Name())
			})
		}
	})
}

func TestRouterBedrockBridgedRemainsDisabled(t *testing.T) {
	t.Parallel()
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("Bedrock forwarding is not supported")
	}))
	t.Cleanup(upstream.Close)
	prov, err := provider.NewBedrock(t.Context(), config.Anthropic{BaseURL: upstream.URL}, config.AWSBedrock{
		Region: "us-east-1", AccessKey: "test-key", AccessKeySecret: "test-secret",
		Model: "test-model", SmallFastModel: "test-small-model",
	})
	require.NoError(t, err)
	router, err := proxy.NewRouter([]provider.Provider{prov}, slogtest.Make(t, nil), nil, noop.NewTracerProvider().Tracer(t.Name()))
	require.NoError(t, err)
	t.Cleanup(router.CloseIdleConnections)
	rec := &routerRecorder{}
	ctx := aibcontext.AsActor(t.Context(), t.Name(), "", nil)
	for _, route := range prov.BridgedRoutes() {
		response := httptest.NewRecorder()
		tasks := utils.NewConcurrentGroup()
		router.Serve(ctx, response, httptest.NewRequest(http.MethodPost, prov.RoutePrefix()+route, nil).WithContext(ctx), rec, tasks)
		require.NoError(t, tasks.Wait())
		require.Equal(t, http.StatusNotFound, response.Code)
	}
	require.Empty(t, rec.starts)
	require.Empty(t, rec.ends)
}

// TestRouterSnapshotsProviders asserts the router routes and reports key
// pools from its own snapshot. Mutating the caller's slice afterwards has
// no effect.
func TestRouterSnapshotsProviders(t *testing.T) {
	t.Parallel()

	pool := testutil.SingleKeyPool(config.ProviderOpenAI, "test-key")
	providers := []provider.Provider{
		keyPoolProvider{Provider: provider.NewDisabledStub("disabled-openai", "openai"), pool: pool},
	}

	router, err := proxy.NewRouter(providers, slogtest.Make(t, nil), nil, noop.NewTracerProvider().Tracer(t.Name()))
	require.NoError(t, err)

	// Replace the caller's entry with a provider the router never saw.
	providers[0] = &testutil.MockProvider{NameStr: "swapped"}

	require.NoError(t, promtestutil.CollectAndCompare(keypool.NewStateCollector(router.KeyPools), strings.NewReader(`
# HELP key_pool_state The number of keys currently in each state (state: valid, temporary, permanent).
# TYPE key_pool_state gauge
key_pool_state{provider="openai",state="valid"} 1
key_pool_state{provider="openai",state="temporary"} 0
key_pool_state{provider="openai",state="permanent"} 0
`), "key_pool_state"))

	resp := httptest.NewRecorder()
	router.Serve(t.Context(), resp, httptest.NewRequest(http.MethodPost, "/disabled-openai/v1/models", nil), nil, nil)
	assert.Equal(t, http.StatusServiceUnavailable, resp.Code)

	resp = httptest.NewRecorder()
	router.Serve(t.Context(), resp, httptest.NewRequest(http.MethodPost, "/swapped/v1/models", nil), nil, nil)
	assert.Equal(t, http.StatusNotFound, resp.Code)
}

// Disabled providers return 503 without reading the body, even if it is oversized.
func TestRouterDisabledProviderOversizedBody(t *testing.T) {
	t.Parallel()

	router, err := proxy.NewRouter(
		[]provider.Provider{provider.NewDisabledStub("disabled-openai", "openai")},
		slogtest.Make(t, nil), nil, noop.NewTracerProvider().Tracer(t.Name()),
	)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/disabled-openai/v1/chat/completions", strings.NewReader("x"))
	req.ContentLength = routing.MaxRequestBodyBytes + 1

	rec := httptest.NewRecorder()
	router.Serve(t.Context(), rec, req, nil, nil)

	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
	assert.Contains(t, rec.Body.String(), routing.ErrorCodeProviderDisabled)
}
