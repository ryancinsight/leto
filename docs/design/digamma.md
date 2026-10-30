# CPU digamma operation

`DigammaOp` is an additive zero-sized `UnaryOp<T: RealScalar>` marker. It
calls Eunomia's single `FloatElement::digamma` scalar implementation. Existing
Leto traversal handles contiguous, strided, offset, transposed and empty views;
this change introduces no scalar formulas, layout copies or new dispatch enum.

The scalar provider's domain contract returns NaN at negative integer poles,
positive infinity at negative zero and positive infinity, negative infinity at
positive zero, and native-precision real values elsewhere. Reduced formats
inherit Eunomia's explicit f32 software-evaluation/storage-narrowing boundary.

Acceptance covers analytic half/integer anchors, transposed input and output,
empty shapes, special-value domains, and native double inputs outside f32
range. Eunomia 0.11 is delivered in PR #185 at merged source `5ae27c75` and
published with matching source receipts. Coeus consumes this operation for
gamma-log reverse mode. Before Leto integration, require final-head review/CI
and the canonical Atlas package planner's post-verification ready set to name
both pending Rust packages. A green wait/empty plan does not establish package
acceptance; Moirai runtime 0.11 is still a registry prerequisite with its owner.
This is CPU coverage only; Hephaestus owns any later device implementation.

The final integration baseline includes main `4a079612` / PR #385's correction
of quadrature `order()` to polynomial degree of exactness. Digamma dispatch,
quadrature coefficients and integration formulas are unchanged by that incoming
fix, while fixed and composite order metadata is observable by consumers.
Final source review, locked API comparison and fresh consumer checks must bind
to the combined revision and that baseline; earlier-base proofs remain historical.
The newer default's publication requires its own registry/source receipts.
