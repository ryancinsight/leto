# ADR 0032: Fused derivative passes write any destination count, over plane windows

- Status: Accepted
- Date: 2026-09-22
- Revised: 2026-09-22 -- decision 4: `combine` receives the destinations'
  held values, so a pass can update a field in place (driver: the kwavers
  elastic step applying its velocity kick inside the acceleration pass).
- Class: [major] public surface; `map_axis_derivatives_triple` is removed and
  `combine` takes the destinations' held values
- Driver: kwavers ADR 133 (elastic stress evaluates in slabs)

## Context

`FiniteDifference3D` fuses axis derivatives and pointwise inputs into one
pass per destination (`map_axis_derivatives`) or per three
(`map_axis_derivatives_triple`). Each pass reads whole grids and writes whole
grids, as one parallel region per call.

Its main consumer, the kwavers elastic step, is a two-stage chain: six
stresses from the displacement gradients, then their divergence. Two costs
followed from the kernel shapes, not from the physics.

- **Destination count became pass count.** Six stresses took four passes --
  the diagonal (three destinations, the triple form) and one per shear --
  reading the displacement nine times and the shear modulus four, in four
  parallel regions. The arity was the runtime's: moirai could split one, two
  or three buffers in lockstep. moirai ADR 0059's K-buffer revision now
  splits any number in one region.
- **Whole-grid intermediates make a DRAM round trip.** At 96 cubed the six
  stress fields are 42 MB against a 36 MB last-level cache, so the divergence
  reads back from DRAM what the stress pass wrote to it. Evaluating the chain
  a slab of x-planes at a time keeps a slab's stresses resident -- but only if
  they live in a buffer reused across slabs. Restricting whole-grid arrays to
  a plane range was measured first: every plane written was last touched a
  whole step earlier, so each line paid a read for ownership and a write-back,
  and the slabs recovered a third of the gap (1.28x at 96 cubed against a 3x
  cache-resident ceiling).

## Decision

1. **One fused pass for any destination count.** `map_axis_derivatives_many`
   takes `[&mut ArrayViewMut3; K]` and a `combine` returning `[T; K]`, as one
   parallel region over moirai's K-buffer unit tasks. `map_axis_derivatives`
   is `K = 1` of it. `map_axis_derivatives_triple` is removed: its callers
   change the name and nothing else. One kernel body -- dense and strided --
   serves every `K`.
2. **Plane windows.** `PlaneWindow` and `PlaneWindowMut` are views holding
   x-planes `first..first + n` of a grid. `map_axis_derivatives_in_windows`
   takes the grid's x-extent and the grid planes to write, with every field,
   pointwise input and destination as a window: a derivative at grid plane `x`
   reads the window's planes around `x` and takes the stencil the grid's
   extent gives `x`. The whole-grid methods are this with whole windows.
3. **Validation names the window at fault.** The destinations hold the same
   planes; each window lies in the grid and shares the destinations' lanes;
   destinations, pointwise inputs and y/z fields hold every plane written;
   x-fields hold every plane the fourth-order stencil reaches (two either
   side, clipped at the grid).
4. **`combine` receives what each destination holds.** Its signature is
   `Fn([T; N], [T; M], [T; K]) -> [T; K]`, the last argument the value each
   destination held before the pass. A result that updates its destination
   -- kwavers' velocity advanced by the acceleration a pass assembles,
   `v + h a` -- reads it there; as a pointwise input it would alias the
   destination it is written to. A combine that replaces its destination
   ignores the argument, and the load is dead once `combine` inlines into
   the kernel.

## Alternatives

- **A quad, quint and sext form beside the triple.** Rejected: arity is a
  variation dimension, and a const generic carries it in one body.
- **Plane ranges over whole-grid arrays only** (this branch's first cut).
  Rejected by measurement: the write-allocate traffic above.
- **Row-granular tasks**, so a short slab still feeds every worker. Measured
  and rejected: interleaved at 64 cubed, plane tasks ran 523-563 us and row
  tasks 591-616 on the whole-grid path; four-row bands did not recover it, so
  per-task overhead is not the mechanism. On windowed slabs the best time did
  not move.
- **An update method beside the replacing one** (decision 4). Rejected:
  replace-or-update is one variation of what `combine` returns, and a
  sibling entry point would carry a second copy of every kernel path.

## Consequences

- kwavers' six stresses take one pass: 169-175 us -> 149-153 at 64 cubed and
  1621-1721 -> 1420-1503 at 96 cubed (release, paired arms), values unchanged
  to the bit.
- kwavers evaluates the acceleration through a reused stress window past the
  cache: 1.5x at 96 cubed and 1.8x at 128 cubed over whole-grid (kwavers ADR
  133 records the measurements and the selection rule).
- kwavers applies its velocity-Verlet kick inside the acceleration pass
  (decision 4), writing each velocity once instead of storing three
  accelerations and reading them back: the whole elastic step runs
  925-968 us against 1028-1056 at 64 cubed, 4452-5436 against 6179-8272 at
  96 cubed and 16041-16782 against 20348-20954 at 128 cubed (release,
  paired), values unchanged to the bit.
- Breaking for `map_axis_derivatives_triple`, whose only caller is kwavers;
  the rename lands in its dependency update.

## Verification

- Every one-cut partition of 1, 2, 3, 5 and 9 planes, and plane-at-a-time,
  rebuilds the whole-grid result bit for bit on the dense and the logical
  walk, single and three destinations; slabs large enough to split across
  moirai tasks do the same.
- Windows holding only the planes a slab reads, into a buffer holding only
  the planes it writes, give the whole-grid planes bit for bit on both walks.
- Six destinations from one pass over nine gradients equal the diagonal pass
  plus three shear passes bit for bit, on both walks.
- A destination updated in place -- `v + m (a + b)` -- equals the composed
  sweeps added to its held value bit for bit on both walks, and a windowed
  update leaves the planes outside its range as they were.
- Each malformed window, destinations holding different planes, and a pass
  with no destination are refused before anything is written, naming the
  fault.
