# Incident: Elevated 500s on Orders API

**Date**: July 10, 2026, 14:32 PDT
**Severity**: SEV-2
**Duration**: ~45 minutes
**Incident Commander**: Avery Chen

## Timeline

| Time | Event |
|------|-------|
| 14:32 | Datadog alert: Error rate > 2% on `/api/v2/orders` |
| 14:35 | Avery acknowledged, started investigating |
| 14:38 | Identified: connection pool exhaustion on `orders-db-primary` |
| 14:42 | Root cause: a long-running analytics query holding connections |
| 14:45 | Killed the runaway query, connections started recovering |
| 14:50 | Sam scaled up connection pool from 20 → 50 as interim fix |
| 15:00 | Error rate back to normal (< 0.1%) |
| 15:17 | All-clear posted in #incidents |

## Root Cause

A scheduled analytics job (Lina's team) ran a full table scan on the orders
table without a statement timeout. It acquired 18 of 20 available connections
and held them for ~12 minutes, starving the API.

## Contributing Factors
- No statement timeout configured on the analytics role
- Connection pool was sized for normal load, no headroom
- No alerting on connection pool saturation (only on error rate)

## Action Items
- [ ] Add statement timeout (30s) for analytics DB role — @lina
- [ ] Increase default connection pool to 50 — @sam
- [ ] Add Datadog monitor for connection pool utilization > 80% — @avery
- [ ] Move analytics queries to read replica — @lina (Q3 goal)
- [ ] Add circuit breaker for connection acquisition — @priya

## Lessons Learned
- We need better isolation between OLTP and analytics workloads
- Connection pool sizing should account for 3x normal load
- The analytics team should run heavy queries on the read replica
