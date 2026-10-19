<a id="adr-0036"></a>

# ADR 0036: Borrowed CSR products separate representation, arithmetic and row execution

Status: Accepted

Date: 2026-10-15. Owner: Codex `/root`.
Board: `LETO-CSR-FORWARD-CONTRACT`. Classification: [arch], implementation
target 0.47.0. The original proposal and its current-API witnesses are
a [patch] documentation/test increment targeting 0.45.3.

## Context

[ADR 0034](0034-layer-boundary-decision.md) gives Leto ownership of narrow CPU
CSR products and Coeus ownership of backend selection. Coeus preservation
[draft #531](https://github.com/ryancinsight/Coeus/pull/531) predates current
fallible tensor/storage contracts. Independently recovered
[draft #533](https://github.com/ryancinsight/Coeus/pull/533) initially removes three copies
from an already compatible unsigned bridge. Its later tensor cutover consumes
the provider capability specified here after resolving those contract differences.

At Leto main `248fe4f63c674f6fca1f89ac19c1f817e67bc417`:

- `CsrView` borrows `usize` indices and requires strictly increasing columns
  in each row. Coeus stores checked `i64` indices, permits duplicates and
  unsorted columns, and accumulates entries in stored order.
- SpMM calls `Scalar::axpy_slice`, which can fuse multiply/add. Coeus performs
  separate native-precision multiplication and addition.
- Sparse products iterate rows serially. Coeus dispatches rows through
  `Backend::parallel_for`, whose unsafe implementor contract guarantees
  exactly-once, in-range calls and synchronous completion.
- `ExecutionStrategy` is a marker. Leto's existing mutable task-partition
  adapter uses a Moirai runtime directly; it cannot retain an arbitrary
  Coeus backend's admission policy through that callback interface.

These are observable differences, not reasons to cast index slices, sort
inputs, add scalar bounds to Coeus, or weaken exact numerical checks.

## Decision

1. Extend the borrowed descriptor with sealed index and ordering policies,
   defaulting to the existing `usize`/canonical contract and retaining the
   current constructor on that exact specialization. Add a checked
   builder for selecting stored-order CSR and `i64` indices. Borrow all three
   original slices; convert individual indices for addressing only after
   nonnegativity, target-width and structural validation.
2. Add a prepared contiguous CSR product, generic over the existing scalar
   and SIMD strategies. The scalar strategy requires only Eunomia's shared
   `NumericElement` contract and uses separate `T` multiplication/addition.
   The SIMD strategy keeps the current `Scalar::axpy_slice` behavior. Neither
   strategy changes entry order. Existing public products retain signatures
   and their arithmetic behavior while delegating to the common row kernel.
3. Validate the complete request before binding output or invoking a worker.
   Separate preparation, mutable-output binding and execution. Workers have
   no fallible input checks and write complete rows, including empty rows.
   The provider accepts initialized `&mut [T]`; no new uninitialized-storage
   API is part of this decision.
4. Expose a narrowly unsafe, synchronous row-dispatch scope over that
   prepared product. Its private-constructor row task contains the provider
   implementation, never a consumer-owned mathematical loop. Coeus adapts
   its unsafe backend guarantee to this scope, first making completion on
   unwind explicit and auditing its implementations. Safe serial entry
   points use the same row task internally. Leto does not select a different
   runtime or threshold for a caller that supplies its backend.
5. Preserve Coeus validation, typed error ordering, allocation and fallible
   mutable access before the adapter. Keep canonical-only structural
   operations restricted to their existing view type. Do not migrate owned
   CSR storage to `u32`, broaden Coeus dense layouts, change accelerator
   backends, or modify either rescue draft in this provider increment. The
   downstream backend safety-contract clarification needs a separately
   reviewed Coeus change before a generic adapter relies on it.

The [co-located design](../../crates/leto-ops/src/application/sparse/README.md)
defines the API, lifecycle, safety obligations, numerical recurrence
and acceptance gates. Implementation and verification status remain tracked
in the checklist and verification record.

## Alternatives

| Alternative | Reason not selected |
| --- | --- |
| Reinterpret `&[i64]` as `&[usize]` | Negatives remain invalid; element widths differ on 32-bit targets. Equal width alone proves neither validation nor portability. |
| Convert indices into new vectors | Sound when checked, but adds two input-sized allocations/copies to every product and leaves the representation gap unresolved. |
| Canonicalize before calling today's view | Sorting and duplicate coalescing change floating-point results. Exact witnesses give 0 before either transform and 1 after it. |
| Select today's SIMD SpMM unconditionally | With epsilon equal to spacing at 1, separate `-1 + (1+epsilon)*(1-epsilon)` produces +0; fusion produces `-epsilon^2`. |
| Call the existing serial product | The prior Coeus comparison measures a 2.069 median time ratio for a 16,384-row SpMV on one Apple M4 Max. It removes backend row parallelism. |
| Schedule Leto task partitions on its Moirai runtime | Safe for that API, but bypasses Coeus backend selection and changes admission/error behavior. |
| Copy a per-row offset array and call full products | Creates per-row descriptors/copies, repeats validation, and still lacks stored-order/signed/arithmetic contracts. |
| Add a sparse executor trait to every Coeus backend | Changes the canonical backend contract and every implementation. The current exactly-once synchronous guarantee already supplies the required scheduler proof. |
| Return `Result` from each worker and use `expect` | Coeus's callback cannot propagate it. Preflight must discharge those checks; panics must not replace typed validation. |

The unsafe dispatch scope is a deliberate public safety boundary. It avoids
per-row allocation, locks, error collection and duplicated kernels. Acceptance
requires both borrow-model Miri checks and negative lifetime/aliasing tests;
the design alone is not evidence that an implementation is sound or fast.

## Verification and delivery order

The [verification record](../verification/csr-forward-contract.md) distinguishes
original-API characterization from executed 0.47 acceptance. Original witnesses
check ordering/canonicalization, scalar versus slice-hook arithmetic, absolute
row offsets, failure atomicity and empty outputs. The final record additionally
checks signed stored order, all scalar families, initialization, dispatch and
checked widths through the real implementation.

Implementation follows the approved provider-first order: checked borrowed representation,
arithmetic policy and common row kernel, validated output/dispatch scope,
then the independently reviewed Coeus adapter. Each slice must satisfy the
full applicable contracts before the next depends on it. The provider gate
includes all Coeus scalar families, 32-/64-bit addressing, complete writes,
zero CSR-copy allocations, exact differential results, strict diagnostics
and paired performance comparisons against the recorded base.

## Sign-off

Ryan Clanton explicitly approves this decision in the task conversation on
2026-10-15 ("approve"), after reviewing the proposal. This satisfies the
global engineering SOP's architectural sign-off requirement and opens the
0.47.0 implementation checklist. Approval covers the provider execution
boundary and the documented downstream safety-contract clarification; it
does not authorize a registry release or merge without checks.


Version adjustment: main `8b4ab407` merges the separately owned numeric-provider
cohort as 0.46.0 during implementation. The additive CSR capability therefore
targets 0.47.0. This changes the version/dependency delivery order, not the
approved behavior or safety boundary. Preserve that cohort and update consumer
provider bindings together under ADR 0029's nominal-identity contract.


The prepared product also encodes known RHS width through a sealed representation
policy. DynamicWidth stores usize; FixedWidth<N> stores unit after checking that
the supplied width equals N. The checked width law preserves the same m*n/k*n
bounds and row recurrence. Existing APIs default to dynamic width; vectors select
fixed width one so callback monomorphization carries neither a runtime width nor
its branches. This addresses the measured narrow-product cost without a second
kernel, unsafe shape assumption or arithmetic-policy change.
