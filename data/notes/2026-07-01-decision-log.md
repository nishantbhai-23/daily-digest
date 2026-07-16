# Platform Team — Decision Log

A record of key technical and process decisions.

---

## Decision 001: Use Redis 7 for Aurora Caching (June 25)

**Context**: Need a distributed cache for Aurora API. Options: Redis, Memcached, DynamoDB DAX.
**Decision**: Redis 7 Cluster
**Rationale**:
- Team already has Redis operational expertise
- Need pub/sub for cache invalidation (Memcached doesn't support this)
- Redis 7 Functions allow server-side scripting for complex invalidation logic
- DAX is too coupled to AWS — we want to stay cloud-agnostic

**Decided by**: Avery, Priya, Sam
**Status**: Approved ✅

---

## Decision 002: Async-first SDK Design (July 2)

**Context**: Internal SDK needs to support both sync and async callers.
**Decision**: Build async-first with sync wrappers
**Rationale**:
- Most internal services are async (FastAPI, async workers)
- Sync wrapper is straightforward (`asyncio.run()` or `loop.run_until_complete()`)
- Going the other direction (sync-first, async wrapper) is much harder
- Matches industry trend (httpx, aiohttp, etc.)

**Decided by**: Avery, Priya
**Status**: Approved ✅

---

## Decision 003: No-Meeting Wednesdays (July 7)

**Context**: Team feedback in retro — too many meetings, not enough focus time.
**Decision**: No recurring meetings on Wednesdays (except incidents)
**Rationale**:
- Developers need at least one guaranteed deep-work day per week
- Research shows context-switching costs ~23 minutes per interruption
- Trial for 4 sprints, then evaluate

**Decided by**: Jordan, Avery
**Status**: Trial ⏳

---

## Decision 004: Feature Flags for Aurora Rollout (July 8)

**Context**: Need safe rollout mechanism for Aurora API v2.
**Decision**: Use LaunchDarkly for feature flags
**Rationale**:
- Already have LaunchDarkly license (Growth team uses it)
- Supports percentage rollouts, user targeting, kill switches
- Better than our homegrown config flags (no audit trail, no gradual rollout)
- Cost: $0 incremental (existing license covers our usage)

**Decided by**: Avery, Sam, Marcus
**Status**: Approved ✅

---

## Decision 005: Migrate to Read Replicas for Analytics (July 10)

**Context**: Analytics queries caused a SEV-2 incident by exhausting the primary DB connection pool.
**Decision**: Route all analytics queries to a dedicated read replica
**Rationale**:
- Complete workload isolation between OLTP and analytics
- Read replica can be scaled independently
- Already have a replica running (just not routed to)
- Prevents future incidents of this class

**Decided by**: Avery, Lina, Sam
**Status**: Planned (Q3) 📋
