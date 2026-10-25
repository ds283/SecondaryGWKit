# Issues from handover prompt 12, §6 (rehearsal R8, the Mac Studio)

None is opened here. The orchestrator decides which go on the `handover` board's §3, and indexes
them in `docs/OPEN_ISSUES.md` if so.

## Proposed for the board's §3

- **[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]** *(found 2026-10-25 by prompt 12
  §6, R8, on the Mac Studio)*. **B1 changed the stored integrals, and the store fingerprint shows
  no difference in `QuadSourceIntegral`.**
  - **What was found.** `compare_with_recorded.py`, R8 against R3, reports one differing class,
    `QuadSource`. That is the value count of its rows, 3 more `QuadSourceValue` rows each.
    `QuadSourceIntegral` has R3's digest. Yet on the one triple both extracts hold, 63 of 96
    `total`s differ from R4's (R3-equal) values, by up to 1.8e-03 of the row set's largest
    |total| (verification §5.4).
    - Format 6's `QuadSourceIntegral.inventory_spec()` is identity only: parents and tags, "no
      validated flag and no value table".
    - Its cross-shard `source` parent resolves to the `QuadSource` row's `reference_digest(key,
      tags)`, which does not include that row's value count.
  - **Why it matters.** The prompt's §6.4 and the orchestrator's §B.1 expected the fingerprint to
    differ in `QuadSourceIntegral`, and they did not anticipate it would not. A fingerprint match
    also does not certify `QuadSourceIntegral` values. That is by the fingerprint's design (an
    identity and count inventory, with "room for computed values later, outside the key", store
    fingerprint audit). But it means a change to how integrals are computed, with unchanged
    `QuadSource` rows, would leave the fingerprint equal. Each R-rehearsal's "equal to 39809dca"
    certified identities and counts, not integral values.
  - **Next step.** None proposed here. Possibly no defect, only the expectation in a prompt to be
    corrected. If values should be covered, that is the store-fingerprint layer's question, and
    any change bumps `FINGERPRINT_FORMAT`.

## Already open, seen again

- `[10-the-readers-draw-their-work-lists-unseeded]`. R8's file counts for `extract_Gk_data`
  (531), `extract_GkSource_data` (160) and `extract_QuadSourceIntegral_data` (76) differ from R7's
  and R3's, with lookups equal. It also limited §6.5's integral comparison to one common triple.
- `[12-realistic-large-x-reads-the-retired-clamp-records]`. Not exercised. Neither script is
  part of R8, per §B.5.
