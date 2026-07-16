# Building a Fair Rate Limiter for Multi-Tenant APIs

*Draft — not for publication*
*Author: Avery Chen | Last edited: July 8, 2026*

---

> **TL;DR**: We built a sliding-window rate limiter that treats enterprise and
> startup customers fairly, without penalizing bursty but legitimate traffic.
> Here's how we designed it, the tradeoffs we made, and what we learned.

## The Problem

At Meridian Labs, our API serves customers ranging from 50-person startups
making 100 requests/day to enterprise accounts pushing 10M+ requests/day.
Our old rate limiter used a fixed-window counter — simple, but it had two
major problems:

1. **Boundary burst**: A customer could send 2x their limit by timing
   requests across window boundaries
2. **One size fits all**: The same 1,000 req/min limit for everyone meant
   enterprise customers were constantly hitting limits while small accounts
   had unused capacity

## Our Approach: Adaptive Sliding Windows

We landed on a hybrid approach:

```
effective_rate = base_rate × tier_multiplier × burst_allowance(recent_history)
```

The key insight: instead of hard limits, we compute a per-customer effective
rate that adapts based on their plan tier AND their recent usage pattern.

### Why not token bucket?

We considered token bucket (and actually prototyped it), but found that:
- It requires persistent state per customer (memory pressure at scale)
- Burst handling is less intuitive to explain to customers
- Our Redis-based sliding window was already battle-tested

## Results

After rolling this out over 2 weeks:
- Rate limit violations dropped 73% for enterprise customers
- Zero increase in abuse or system overload
- Customer satisfaction (NPS) for API experience went from 34 → 52

## Lessons Learned

1. **Talk to your customers first** — We interviewed 8 customers before
   writing a single line of code. Three of them had workarounds that were
   more complex than our entire rate limiter.
2. **Monitor the monitors** — Our rate limiter itself became a reliability
   concern. We added circuit breakers to fall back to a simple fixed window
   if Redis latency spikes.
3. **Docs matter more than code** — The biggest impact came from clearly
   documenting the rate limits in our API docs with examples.

---

*TODO: Add architecture diagram*
*TODO: Get review from Priya and Marcus before publishing*
*TODO: Add code samples in Python and JavaScript*
