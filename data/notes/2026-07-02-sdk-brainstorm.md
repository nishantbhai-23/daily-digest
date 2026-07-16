# Platform SDK — Design Brainstorm

*Scratch notes from brainstorm session, July 2, 2026*
*Attendees: Avery, Priya, Nina, Ben*

## Goal
Build an internal SDK that other teams at Meridian can use to integrate with
the Platform API without dealing with raw HTTP, auth, retries, etc.

## Key Questions
- What languages? Python first, then TypeScript?
- Sync vs async? Both? Async-first with sync wrapper?
- How do we handle versioning? Semver? API version pinning?
- Should the SDK be open-source eventually?

## Ideas

### Developer Experience First
- Make the "hello world" < 5 lines of code
- Auto-discovery of API endpoints from OpenAPI spec
- Built-in retry with exponential backoff + jitter
- Rich error types (not just HTTP status codes)

```python
# Dream API:
from meridian import Client

client = Client()  # auto-discovers credentials
orders = client.orders.list(status="active", limit=50)

for order in orders:
    print(order.id, order.customer.name)
```

### Observability Built In
- Structured logging with request IDs
- OpenTelemetry traces out of the box
- Metrics: request count, latency histograms, error rates

### Testing Support
- Mock client for unit tests
- Record/replay mode for integration tests
- Fixtures generator from OpenAPI spec

## Architecture Options
1. **Code-gen from OpenAPI** — pros: always up-to-date; cons: generated code can be ugly
2. **Hand-written with spec validation** — pros: beautiful DX; cons: maintenance burden
3. **Hybrid** — generate the transport layer, hand-write the public API

→ Leaning toward option 3. Let's prototype next week.

## Next Steps
- [ ] Avery: prototype the hybrid approach (1 endpoint)
- [ ] Nina: draft SDK documentation structure
- [ ] Ben: survey how customers currently integrate (API patterns)
- [ ] Priya: evaluate code-gen tools (openapi-generator vs custom)
