# Structural Operations

Structural operations change shape or partition logical regions. They are
implemented against layouts and storage contracts, so their copy behavior is
part of the API.

## Concatenation and padding

`concat` joins arrays along an existing axis. All non-joined axes must match;
the joined extent is the sum of the inputs. `pad` adds a configured width on
each side of an axis and fills the new region with the requested value. Both
operations allocate a C-contiguous result because the output combines data
from multiple logical regions.

## Split and stack

`split` partitions an array along an axis into read-only views. The returned
views share the original storage, so splitting is zero-copy and the source
must outlive every view. The split boundaries are validated before any view is
returned.

`stack` inserts a new axis and copies the inputs into one C-contiguous result.
The `InsertAxis` type-level helper expresses the rank change from `N` to
`N + 1`; callers cannot accidentally treat the result as the original rank.

Choose `split` when ownership can remain with the source. Choose `concat`,
`pad`, or `stack` when a new independent buffer is the intended result. The
distinction matters for memory traffic and for whether later mutation is
allowed.
