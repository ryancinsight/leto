# Current Python binding provider [minor]

Continue Leto362 on owner `b44b9f07b0fefab0f576ba971051514aca1c8ad6`.
Its move from `b994b32cd8a54896665be603fa21f64c037d903e` changes backlog only;
all tested runtime, manifest and lock bytes are preserved. Retain all approved
CSR changes, current Moirai0.10 and merged linkage/Mach-O/wheel-contract fixes.
The earlier local duplicate packaging edits are dropped. No owner ref is written.

The untouched owner head with PyO3.23.5 rejects Python3.14 in a neutral-directory
locked control with interpreter checks enabled; the archived diagnostic names
its maximum supported3.13. Updating only requirements
to current0.30 reproduces missing `with_gil`, `allow_threads` and legacy Python
initialization APIs. Replace those calls with documented `attach`, `detach` and
`Python::initialize`, preserving each closure's calculation and assertions.
Explicit `gil_used=true` retains the old module's GIL-required contract after
PyO3 changed its default; free-threaded safety is not asserted. Remove the old
ABI3 forward-compatibility environment suppression so interpreter checks run.
Current resolved dependencies are PyO3.30.1 and NumPy.30.0. No native kernel,
core/ops API, CSR source, limit, assertion or release permission changes.

The independently versioned binding proposal advances published0.46.1 to0.47.0.
Core/ops stay on the owner's reserved0.47.0. Rust extension consumers must
coordinate the nominal PyO3.30 type identity; Python operation names, dtype/error
contracts and Python3.9 interpreter floor remain. Extend the existing guarded wheel
matrix with3.14, preserving all owner verification/attestation/publisher guards.

## Verification

- Fresh normal lock resolves seven canonical provider repositories, each once.
- All1,223 workspace/all-feature native tests pass:1,202 unchanged core/ops and
  21 binding tests, zero skips; existing deadlines remain.
- Strict all-target/all-feature workspace Clippy, formatting,38 doctests plus
  one existing ignore, warning-denied docs and actual Rust1.95 workspace check pass.
- Ordinary locked optimized Maturin wheels load without manual binary edits.
  All39 mandatory tests pass on CPython3.14.7 and3.9.25, zero skips:29 original
  NumPy/SciPy cases, nine owner wheel contracts and one new32-call/4-thread
  computed-value regression. Both wheel payloads have no libpython linkage.
- Python3.9 uses its latest compatible test-only NumPy/SciPy packages; production
  Rust dependency requirements remain current. No interpreter-version bypass.
- Prior feature-disabled1,190 cases belong to the earlier unchanged core cohort;
  this record does not relabel them as a new final-head minimal-feature run.

Core/ops Rust source is byte-identical to the owner, so its explicit196-check
API comparisons remain source-scoped evidence; final combined exact-head review
and hosted full-platform wheel matrix are required. Existing memory/performance
receipts are revision-scoped, not new acceptance. No merge, tag, package upload,
credential change or service action occurs. The built wheels are local artifacts.

## Pending Aequitas 0.4 boundary [minor]

[Aequitas141](https://github.com/ryancinsight/aequitas/pull/141) proposes law0.4.0
and binding0.2.0; implementation8c50883a01516aab8a22323d817cb8f27f59b307
is independently reviewed separately. ADR0020 acceptance, owner140 version
coordination and hosted/free-threaded/book gates remain pending. Current0.3 is
the verified public cohort for this checkpoint, not a permanent older-version
workaround. Do not consume an unaccepted candidate or edit its provider sources.

Static inspection of Harmonia, Athena119, Leto362 and Hephaestus497 finds no
generic `Quantity::powi<P>` or `TaggedDimension` producer requiring bound
propagation. Harmonia's `FieldEnvelope<T,D,RANK>` borrows quantities and does
not perform dimension powers or emit runtime tags. Leto and Hephaestus stencil
contracts extract concrete Length/Meter values, then use scalar `powi(i32)`;
that operation does not gain the proposed type-level bounds. Athena contains
neither surface. This is scoped source inspection, not a future-provider build.

After acceptance, renew all direct and transitive Aequitas requirements together
(including Horae and Leto/Hephaestus typed-stencil providers), regenerate fresh
locks and repeat typed quantity/time/spacing consumer and API gates against the
accepted exact target. Generic quantity-power producers, if introduced, need
`P: Integer + IsGreaterOrEqual<N2147483648,Output=B1> +
IsLessOrEqual<P2147483647,Output=B1>`; generic dimension-tag producers need each
axis in[-128,127]. Existing `D: TaggedDimension` consumers keep that bound.
