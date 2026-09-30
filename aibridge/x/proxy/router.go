// Package proxy provides stateless AI Gateway request routing.
package proxy

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"slices"

	"go.opentelemetry.io/otel/trace"
	"golang.org/x/xerrors"

	"cdr.dev/slog/v3"
	"github.com/coder/coder/v2/aibridge"
	"github.com/coder/coder/v2/aibridge/config"
	"github.com/coder/coder/v2/aibridge/keypool"
	"github.com/coder/coder/v2/aibridge/metrics"
	"github.com/coder/coder/v2/aibridge/provider"
	"github.com/coder/coder/v2/aibridge/recorder"
	"github.com/coder/coder/v2/aibridge/routing"
	"github.com/coder/coder/v2/aibridge/utils"
)

// Router serves the AI Gateway routes of an immutable snapshot of providers.
// It holds no per-user or per-request state, so a single instance serves every
// actor and is replaced wholesale when the provider configuration changes.
//
// Router has no concept of authentication or authorization; the caller is
// responsible for authenticating requests before they reach the router.
//
// Router is safe for concurrent use.
type Router struct {
	mux *http.ServeMux

	// providers is an owned copy of the provider list, fixed at construction.
	providers  []provider.Provider
	transports []*http.Transport
}

// forwardingHandlerFunc adapts recorder-aware handlers for ServeMux registration.
type forwardingHandlerFunc func(context.Context, http.ResponseWriter, *http.Request, recorder.Recorder, *utils.ConcurrentGroup)

func (f forwardingHandlerFunc) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f(r.Context(), w, r, nil, nil)
}

// NewRouter creates a [*Router] for the given providers, whose
// names must be valid and unique.
//
// Each configured-but-disabled provider serves a 503 sentinel on every path
// under its name. Passthrough routes use the existing passthrough handler.
// Bridged routes require an actor on the request context and a recorder supplied
// to Serve. Bedrock bridged routes remain unsupported.
func NewRouter(providers []provider.Provider, logger slog.Logger, m *metrics.Metrics, tracer trace.Tracer) (_ *Router, outErr error) {
	if err := provider.ValidateProviders(providers); err != nil {
		return nil, err
	}

	snapshot := slices.Clone(providers)
	mux := routing.NewProviderMux(snapshot, logger)
	router := &Router{mux: mux, providers: snapshot}
	defer func() {
		if outErr != nil {
			router.CloseIdleConnections()
		}
	}()
	for _, prov := range snapshot {
		if !prov.Enabled() {
			continue
		}

		bridged := http.NotFoundHandler()
		if prov.Type() != config.ProviderBedrock && len(prov.BridgedRoutes()) > 0 {
			handler, transport, err := newForwardingHandler(prov, logger, m, tracer)
			if err != nil {
				return nil, err
			}
			bridged = forwardingHandlerFunc(handler.serve)
			router.transports = append(router.transports, transport)
		}
		for _, path := range prov.BridgedRoutes() {
			route, err := url.JoinPath(prov.RoutePrefix(), path)
			if err != nil {
				return nil, xerrors.Errorf("configure provider %q bridged route: %w", prov.Name(), err)
			}
			mux.Handle(route, bridged)
		}

		passthrough := aibridge.NewPassthroughRouter(prov, logger.Named(fmt.Sprintf("passthrough.%s", prov.Name())), m, tracer)
		for _, path := range prov.PassthroughRoutes() {
			route, err := url.JoinPath(prov.RoutePrefix(), path)
			if err != nil {
				return nil, xerrors.Errorf("configure provider %q passthrough route: %w", prov.Name(), err)
			}
			mux.Handle(route, http.StripPrefix(prov.RoutePrefix(), passthrough))
		}
	}

	return router, nil
}

// Serve forwards a request with its authenticated recorder. Bridged routes
// register asynchronous processing in tasks before returning; the caller must
// drain tasks before releasing the recorder and processCtx. Forwarding uses
// r.Context(), independently of processCtx. Passthrough routes need neither rec
// nor tasks.
func (p *Router) Serve(processCtx context.Context, rw http.ResponseWriter, r *http.Request, rec recorder.Recorder, tasks *utils.ConcurrentGroup) {
	// Cap the body as it is read; routes that do not read it retain their status.
	if r.Body != nil && r.Body != http.NoBody {
		r.Body = http.MaxBytesReader(rw, r.Body, routing.MaxRequestBodyBytes)
	}
	handler, pattern := p.mux.Handler(r)
	if forwarding, ok := handler.(forwardingHandlerFunc); ok && r.RequestURI != "*" {
		r.Pattern = pattern
		forwarding(processCtx, rw, r, rec, tasks)
		return
	}
	p.mux.ServeHTTP(rw, r)
}

// KeyPools returns the non-nil key pools from this router's provider snapshot.
func (p *Router) KeyPools() []*keypool.Pool {
	return provider.CollectKeyPools(p.providers)
}

// CloseIdleConnections releases idle connections owned by bridged forwarding.
// Active requests retain their connections until they finish.
func (p *Router) CloseIdleConnections() {
	for _, transport := range p.transports {
		transport.CloseIdleConnections()
	}
}
