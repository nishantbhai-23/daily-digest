# SRE Capacity Crunch — Quick Retro

## What happened
An unrelated on-call fire ate two days out of the sprint meant for the
Lone Star Power forecasting API fix, pushing it right up against the
customer's patience. Shipped Friday, but closer than it should have been.

## Contributing factors
- On-call rotation has effectively been Felix + Layla for the last month,
  not the full rotation
- No dedicated SRE headcount — Q3 req just opened, won't help until it's
  filled
- No slack built into sprint planning for on-call interruptions

## Follow-ups
- [x] Ship the forecasting API fix (done Friday)
- [ ] Get the senior SRE req staffed and interviewing
- [ ] Rebalance the on-call rotation to include Theo's team
- [ ] Add explicit on-call buffer to next sprint's planning instead of
      treating interruptions as free capacity
