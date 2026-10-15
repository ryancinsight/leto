# Borrowed CSR forward product design

Accepted under [ADR 0036](../../../../../docs/adr/0036-borrowed-csr-product-contract.md).
Ryan Clanton approves the decision on 2026-10-15. Checked representation lives
in `csr/`; arithmetic, preparation and the common row task live in `product/`.
The existing `spmv.rs` and `spmm.rs` entry points delegate to that kernel.

## Ownership and scope

Leto owns CSR validation, address decoding, the row recurrence and output
writes. Coeus owns tensors, typed backend errors, allocation, layout rejection
and backend selection. The integration removes Coeus's forward mathematical
loops only after the provider proves the contracts below. Backward kernels,
owned CSR index-width changes, noncontiguous Coeus operands and accelerators
remain separate work.

## Representation and public shape

The descriptor is `CsrView<'a, T, I = usize, O = Canonical>`. `I`
implements a sealed `CsrIndex` role, initially for `usize` and `i64`. `O`
implements a sealed `CsrOrdering` role:

| Policy | Structural requirement | Numerical meaning |
| --- | --- | --- |
| `Canonical` (default) | Each row has strictly increasing column indices | Existing unique, sorted CSR semantics |
| `StoredOrder` | Columns may repeat or decrease | Every stored entry participates in its physical order; no sorting, coalescing or zero dropping |

Keep `CsrView::from_slices` on the exact existing `usize`/`Canonical`
specialization so existing unannotated constructor calls and integer literals
still infer it. A single checked builder,
`CsrView::builder(values, columns, offsets, rows, columns_count)`, starts with
`Canonical`; `with_order::<StoredOrder>()` selects the ordering type and
`build()` validates it. Index type infers from the borrowed index slices.
One validator handles both policies; the strict-order check alone varies.
Do not add constructors named after integer widths or duplicate validators.

`CsrMatrix<T>` retains owned `usize` canonical storage. Its `as_view()` retains
the current return type. `CscMatrix::from_csr`, structural algorithms and
owned mutation continue accepting their current canonical type; stored-order
views cannot silently enter algorithms that rely on uniqueness. The existing
`LETO-CSR-INDEX-WIDTH-2026-09-27` proposal is independent.

Before constructing any view, prove:

1. `m + 1` is representable, `offsets.len() == m + 1`, and
   `columns.len() == values.len() == q`.
2. Each index decodes losslessly into the target's `usize`. In particular,
   reject negative `i64` values and positive values above `usize::MAX`.
3. `offsets[0] == 0`, `offsets[m] == q`, and offsets are nondecreasing and
   bounded by `q`. Each column is less than the shared dimension `k`.
4. Enforce strict column order only under `Canonical`.

All slices retain their original pointers and element types. Validating once
proves subsequent private scalar address conversions infallible; a narrowing
conversion is justified by that proof, never by equal pointer widths. There
is no bit reinterpretation, converted index array, unchecked public validator
or caller assertion that can manufacture a validated view. Empty dimensions
do not bypass validation. Complexity is `Theta(m + q)`, auxiliary space `O(1)`.

## Preparation, output binding and execution

The `CsrProduct<'a, T, I, O, A>` stores borrowed validated CSR parts,
a contiguous dense slice of exact shape `k * n`, and its checked dimensions.
`A` implements a sealed operation-specific `CsrArithmetic<T>` role. Use
`n = 1` for SpMV; SpMM uses the dense RHS column count. This is one CSR row
traversal, with arithmetic policy selecting the row update.

| Operation | Result and obligation |
| --- | --- |
| `CsrProduct::<..., A>::prepare(view, rhs, rhs_columns)` | `leto::Result<Self>`; check `k*n`, `m*n`, and exact RHS length without writing or allocating |
| `product.bind_output(&mut [T])` | `leto::Result<CsrOutput<...>>`; check exact output length and retain exclusive borrow |
| `product.execute(&mut [T])` | Safe serial convenience operation using the same binding and row task |
| `unsafe bound.dispatch_with(dispatch)` | Consume the output binding; invoke `dispatch(row_count, row_task)` synchronously and return the callback's result |
| `unsafe row_task.run(row)` | Infallible complete write of one row; no input checks, allocation, or executor selection |

The row task has private fields and a private constructor. It is a copyable
pointer descriptor with `Send`/`Sync` only under the proven scalar/index
requirements. It owns no buffer. The binding retains the lifetimes of inputs
and output; the scope temporarily erases them only for the existing `'static`
Coeus callback. Both dispatch and task invocation are unsafe because a
consumer could otherwise retain a task, call one row concurrently twice, or
pass an out-of-range index. No raw pointer is exposed as a safe capability.

The caller of `dispatch_with` must prove:

- On success, each row in `0..m` is invoked exactly once, and no other row
  is invoked. On error or panic, an arbitrary subset may have run, each at
  most once; all admitted work must still complete.
- All callbacks, including unwinding callbacks, finish before dispatch returns
  or unwinds. No task copy or captured pointer is used after that scope.
- The exclusively borrowed output is inaccessible through any other alias
  until the scope ends; shared CSR/RHS data stay immutable and alive.
- `run(r)` calls on different workers address distinct rows. Recursive or
  concurrent invocation of the same row is prohibited.

The provider proves from checked `m*n` and output length that row `r` occupies
`[r*n, (r+1)*n)`. For `r != s` and `n > 0`, these intervals are disjoint.
For `n == 0`, no references to an output element are formed and no writes
occur. Every row accumulator starts at `T::ZERO`; a vector uses a single local
accumulator and writes its completed value once. Wider rows initialize the
destination before accumulating. No old output value enters the recurrence. Shared input
and output slices obey Rust's aliasing rules before pointer erasure.

Coeus's current unsafe `Backend` documentation states exactly-once,
in-range, synchronous dispatch on normal return. It does not state completion
on unwind explicitly enough for an arbitrary external implementation. Before
adding the generic adapter, a separately reviewed Coeus change must make that
safety obligation explicit, audit every implementation and test real backend
panic completion. This is an implementor-contract change even though the
method signature remains unchanged; document it under Breaking in that
pre-1.0 minor release. The current Sequential implementation runs inline;
the locked Moirai indexed implementation catches task panics and drains its
scope before returning an error. That source evidence does not substitute for
the public trait obligation or an executed integration regression.

The Coeus adapter then contains the safety proof and a call to
`row_task.run(r)`, with no arithmetic or CSR pointer math. No new generic
executor trait, Moirai runtime selection or threshold belongs in the provider
product.

All preparation/binding failures precede dispatch and leave output unchanged.
After preflight there are no recoverable shape/index/layout failures inside a
worker. Arithmetic panic behavior remains the selected scalar's behavior;
arithmetic is not fallible by this API. A fallible executor may return an
error after partial output writes, but must still join every admitted task.
The initialized output remains valid; Coeus must never return that tensor as
a successful product. Coeus's present `parallel_for` returns unit.

## Arithmetic and scalar bounds

For each row `r`, column `j`, and stored offset `p` in increasing order from
`offsets[r]` to `offsets[r+1]`, the scalar strategy implements exactly:

```text
s_0 = T::ZERO
product_p = values[p] * rhs[columns[p], j]   // T's multiplication
s_(p+1) = s_p += product_p                  // T's AddAssign
output[r,j] = s_end
```

The strategy requires Eunomia `NumericElement`, which is already a supertrait
of Coeus `Scalar`. It must not require Leto's slice-hook `Scalar` extension.
Preserve all current Coeus signed/unsigned integer widths, `f32`, `f64`,
`F16`, `Bf16`, and `Complex<T>` for the existing four real float types. Execute
operators in the declared type; preserve the existing scalar's own documented
implementation, overflow and exceptional-value behavior. Do not bulk widen,
change an accumulator type, skip explicit zeros, fuse, reorder or reassociate.

The SIMD strategy requires Leto `Scalar` and uses `axpy_slice`. Existing
SpMM chooses it, including when the RHS has width one; existing SpMV chooses
the separate scalar recurrence, as today. Coeus chooses the scalar strategy
for both products. Strategy selection is static, zero-sized and inlined;
auto-vectorization across independent RHS columns is allowed only when it
preserves each output's operation sequence and rounding.

The stored CSR identity proves the real-arithmetic formula, but does not
justify floating-point transformations. In binary64, values
`[2^53, 1, -2^53]`, columns `[1, 0, 1]` and unit RHS give +0; coalescing gives
1. Sorting columns `[1, 1, 0]` for those values also changes +0 to 1. With
`e = 2^-52`, separate `-1 + (1+e)*(1-e)` gives +0 while FMA gives `-2^-104`.
These exact witnesses forbid an approximate-equivalence substitution.

## Dense storage, errors and initialization

The new product accepts contiguous slices. Coeus first preserves its current
CSR validation, dense rank/shape checks, noncontiguous-layout rejection and
readable-storage checks. It slices the validated logical region at its
nonzero offset; it never passes the whole backing allocation as the RHS.
The provider does not broaden Coeus's layout contract by materializing views.
Existing Leto public view products retain their strided-input materialization.

Keep Coeus's error categories and ordering: negative indices return
`InvalidNumericInput`; invalid or non-representable columns return
`IndexOutOfRange` (an unrepresentable column reports `usize::MAX` as its
index). Non-representable row offsets also return `InvalidNumericInput`;
the existing message calls them negative even when the original `i64` was
positive. A representable offset outside the stored range returns `Storage`.
Preserve this width-dependent distinction during integration; correcting its
wording/category is a separate behavioral decision. Shape-count overflow
retains `Overflow`. Check current validators for the precise precedence when
several faults coexist. Provider constructors/preparation return existing
`LetoError` categories; no new public error-enum variant is required. The
adapter maps unexpected provider failures explicitly to contextual backend
errors instead of `expect`, panic or error masking.

Retaining Coeus validation means provider construction may validate indices a
second time. Measure that cost in the complete product; do not add an unsafe
unchecked-construction escape solely to remove it. A later shared-validation
API would need its own compatibility and error-order proof.

Coeus keeps `unsafe Tensor::alloc_on(...) ?` and fallible
`tensor.as_mut_slice()?`. Current CPU storage initializes an uninitialized
allocation before forming that mutable typed slice (`raw_slice_mut_cow`).
The provider therefore receives valid initialized `T` storage. Preserve this
mechanism and propagate its errors; do not form `&mut [T]` directly over
uninitialized memory or read the old destination to accumulate. Any future
write-only `MaybeUninit<T>` API is separate, measured work.

For `m == 0` or `n == 0`, the checked output is empty. For `k == 0`, valid CSR
has no entries and every output is `T::ZERO`. For an empty row at positive
`k`, the row's entire output is also `T::ZERO`. Successful execution writes every `m*n` slot. Accumulation reads only an
initialized local or destination accumulator, never the old destination value.
No zero-width chunk iterator, pointer dereference or dense row read is needed
when `n == 0`.

## Acceptance and migration

ADR sign-off is recorded. Required evidence is enumerated in the
[verification record](../../../../../docs/verification/csr-forward-contract.md).
Provider implementation must cover checked signed borrowing, strict versus
stored ordering, all scalar families, failure atomicity, offset slices,
empty dimensions, both borrow models and architecture portability before the
Coeus forward loops are removed. The Coeus integration keeps its public
`Result<Tensor<T, B>, B::Error>` signatures and backend trait unchanged.

Measure both provider and complete Coeus products, including double
validation, initialization, scheduler and all allocation costs, against the
exact recorded main revision. Use the existing paired SpMV/SpMM workloads
and include wide outputs beyond the old inline row-scratch capacity. Check
every output before timing, consume results, and retain the same input domain.
If separate-rounding SpMM costs more than fused SpMM, that comparison does
not override the Coeus baseline contract. Resolve any regression against
Coeus's unchanged baseline before claiming the full cutover complete.


## Known RHS width and callback specialization

The complete-graph timing repeat finds a 2–8% SpMV cost. Forced adapter
inlining brings small-vector medians near parity but leaves about 6% in the
large Moirai case. Native assembly retains a shared dynamic-width adapter and
width-dependent callback branches. A vector's RHS width is statically one.

A sealed `ProductWidth` representation policy records that fact without a
second recurrence: `DynamicWidth` stores usize; `FixedWidth<N>` stores unit.
Preparation checks a supplied width against N before accepting the product.
All existing product types default to DynamicWidth, preserving their API and runtime
width representation. Tasks retain the checked policy's value; fixed-width instantiations
therefore carry no runtime width and expose N to callback monomorphization.

The proof extends product preparation: `checked(n)` implies
`value(metadata) == n`. The existing checked m*n/k*n and disjoint-row arguments
then apply unchanged. A mismatch returns typed ShapeMismatch before output
binding or writes. Width zero retains its no-reference/no-read behavior. The
existing arithmetic policy remains independent, including legacy SIMD rounding.
Alternatives are retaining measured overhead or adding separate vector math;
source duplication does not satisfy the provider ownership contract. Verify
fixed widths 0/1/3, mismatch rejection, all scalar families, both borrow models,
32-bit provider execution, old API compatibility and the unchanged paired
performance workloads before accepting the optimization.


Signed row validation checks representability and monotonicity in one pass.
A deferred monotonicity flag preserves the existing error order: any invalid
representation wins, then endpoints, monotonicity and column bounds. With
nonnegative decoded offsets, the initial lower bound zero is automatic; the
flag is equivalent to all adjacent offsets being ordered. Endpoints 0/nnz then
prove the complete offset bound. Multi-fault witnesses characterize that order
before fusion, so fewer serial index reads do not change diagnostics.
