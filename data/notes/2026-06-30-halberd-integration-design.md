# Halberd Integration — Technical Design Notes

**Author**: Priya Iyer
**Status**: In progress
**Reviewers**: Avery, Jordan, Kenji

## Summary
Halberd's ERP exports shipment and inventory data nightly via SFTP. We need
to ingest, normalize, and reconcile it against our own inventory model, then
surface supplier risk alerts within 15 minutes of ingestion.

## Open Question: Multi-Warehouse
Halberd operates 2 warehouses today, possibly 3 if the Ohio plant conversation
(via Tomás) goes anywhere. Our current data model assumes single-warehouse per
customer. Two options:

1. **Workaround for Halberd only** — hacky per-customer flag, ships faster
2. **Proper multi-warehouse support** — bigger lift, unblocks Northstar/Ohio too

Leaning toward option 2 given Northstar is asking similar questions, but it
pushes the Halberd deadline by ~1 week.

## Reconciliation Logic
- Match on SKU + warehouse + timestamp window
- Flag mismatches > 5% as anomalies for the supplier risk alert
- Nightly batch for now, streaming is a v2 goal

## Timeline
- Week 1: SFTP ingestion + normalization
- Week 2: Reconciliation engine + anomaly detection
- Week 3: Multi-warehouse support (if we go with option 2)
- Week 4: Beta with Halberd ops team

## Open Questions
- [ ] Multi-warehouse: option 1 or 2? (Avery to decide with Priya/Jordan)
- [ ] Do we need SLA guarantees written into the Halberd MSA for ingestion latency?
