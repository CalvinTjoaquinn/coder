package proxy_test

import (
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
	"github.com/coder/coder/v2/aibridge/internal/testutil"
	"github.com/coder/coder/v2/aibridge/keypool"
	"github.com/coder/coder/v2/aibridge/metrics"
	"github.com/coder/coder/v2/aibridge/provider"
	"github.com/coder/coder/v2/aibridge/routing"
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
		{name: "EnabledBridgedRoute", path: "/openai/v1/chat/completions", wantStatus: http.StatusNotFound, wantBody: "404 page not found"},
		{name: "EnabledPassthroughRoute", path: "/openai/v1/models", wantStatus: http.StatusOK, wantBody: "upstream"},
		{name: "UnknownEnabledRoute", path: "/openai/admin", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
		{name: "UnknownProvider", path: "/unknown/v1/models", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
		{name: "Root", path: "/", wantStatus: http.StatusNotFound, wantBody: "route not supported"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, tc.path, nil))

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
		router.ServeHTTP(response, req)
		if path == "/messages" || path == "/models/bridged" {
			require.Equal(t, http.StatusNotFound, response.Code)
			require.Equal(t, "404 page not found\n", response.Body.String())
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
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/disabled-openai/v1/models", nil))
	assert.Equal(t, http.StatusServiceUnavailable, resp.Code)

	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/swapped/v1/models", nil))
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
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
	assert.Contains(t, rec.Body.String(), routing.ErrorCodeProviderDisabled)
}
