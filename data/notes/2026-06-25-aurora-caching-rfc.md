# RFC: Caching Strategy for Project Aurora API

**Author**: Avery Chen
**Status**: Draft
**Created**: June 25, 2026
**Reviewers**: Priya Sharma, Sam O'Brien, Jordan Reeves

## Summary

This RFC proposes a multi-layer caching strategy for the Aurora API to reduce
latency and database load. The goal is to achieve P95 latency < 100ms for
read-heavy endpoints (currently ~350ms).

## Background

The Aurora API serves ~2M requests/day with a read:write ratio of 85:15.
Current architecture queries PostgreSQL directly for every request. With the
upcoming Enterprise launch, we expect 5x traffic growth by Q4.

## Proposed Architecture

### Layer 1: Application-level cache (in-process)
- **Technology**: LRU cache with TTL (Python `cachetools`)
- **Scope**: Per-instance, hot data only
- **TTL**: 30 seconds
- **Invalidation**: TTL-based (eventual consistency acceptable)

```python
from cachetools import TTLCache

cache = TTLCache(maxsize=1000, ttl=30)

def get_order(order_id: str) -> Order:
    if order_id in cache:
        return cache[order_id]
    order = db.query(Order).get(order_id)
    cache[order_id] = order
    return order
```

### Layer 2: Distributed cache (Redis)
- **Technology**: Redis 7.x cluster
- **Scope**: Shared across all API instances
- **TTL**: 5 minutes
- **Invalidation**: Write-through on mutations + pub/sub for cross-instance

### Layer 3: CDN edge caching (future)
- For public/semi-public endpoints only
- Deferred to Phase 2

## Metrics & Success Criteria
| Metric | Current | Target |
|--------|---------|--------|
| P50 latency | 180ms | < 50ms |
| P95 latency | 350ms | < 100ms |
| P99 latency | 800ms | < 250ms |
| DB queries/sec | 4,200 | < 1,000 |
| Cache hit rate | N/A | > 85% |

## Risks & Mitigations
1. **Stale data** — Mitigated by short TTLs and write-through invalidation
2. **Cache stampede** — Use probabilistic early expiration
3. **Redis failure** — Graceful fallback to DB-only mode

## Timeline
- Week 1: Redis cluster setup + L2 cache implementation
- Week 2: L1 cache + invalidation logic
- Week 3: Load testing + monitoring dashboards
- Week 4: Staged rollout (canary → 10% → 50% → 100%)

## Open Questions
- [ ] Should we use Redis Cluster or Redis Sentinel?
- [ ] Do we need cache warming on deploy?
- [ ] What's the budget for Redis infrastructure?
