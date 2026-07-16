# Project Aurora — Launch Checklist

**Target launch**: July 21, 2026
**Owner**: Avery Chen
**Stakeholders**: Jordan (Eng), Marcus (PM), Elena (Design), Derek (QA)

## Pre-Launch

### Engineering
- [x] API v2 endpoints implemented and tested
- [x] Rate limiter deployed to staging
- [ ] Caching layer (L1 + L2) implemented
- [ ] Load testing complete (target: 10K req/sec)
- [ ] Runbook written for on-call team
- [ ] Feature flags configured for staged rollout
- [x] Database migrations tested on staging

### Security
- [x] Security review completed with Chris
- [ ] Penetration test scheduled
- [ ] API keys rotation mechanism verified

### Documentation
- [ ] API reference updated (Swagger/OpenAPI)
- [x] Internal architecture docs in Notion
- [ ] Customer-facing migration guide
- [ ] Changelog entry drafted

### QA
- [x] Integration tests passing (142/142)
- [ ] Performance regression suite green
- [ ] Manual exploratory testing by Derek's team
- [ ] Beta customer feedback addressed

## Launch Day
- [ ] Feature flag: enable for 10% of traffic
- [ ] Monitor error rates, latency, CPU for 1 hour
- [ ] If green: ramp to 50%, then 100%
- [ ] Send customer announcement email
- [ ] Post in #engineering Slack channel

## Post-Launch
- [ ] Monitor for 48 hours
- [ ] Collect customer feedback
- [ ] Schedule postmortem if any incidents
- [ ] Plan v2.1 iteration based on feedback
