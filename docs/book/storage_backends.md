# Storage Backends

The storage parameter of `Array<T, S, N>` makes allocation and ownership
explicit. Every backend implements the narrow `Storage` contract; mutable
backends additionally implement `StorageMut`.

## Owned storage

`VecStorage<T>` owns a heap allocation and is the default for dynamic arrays.
Use it when the array must outlive the input or when a result is built by a
kernel. `StackStorage<T, CAP>` stores a fixed-capacity buffer inline and is
appropriate when the capacity is a compile-time structural property. The
capacity is a const generic, so the storage cannot silently grow past its
declared bound.

## Borrowed storage

`SliceStorage<'a, T>` and `SliceStorageMut<'a, T>` expose an existing slice as
array storage. They do not allocate or copy. The lifetime on the storage
connects every derived array or view to the source slice, and mutable storage
inherits Rust's exclusive-borrow rule.

## Mnemosyne integration

`MnemosyneStorage<T>` is optional and routes allocation through the Atlas
memory provider. It is enabled by the `mnemosyne-memory` feature because it
adds an infrastructure dependency; the core layout and view contracts remain
unchanged. Consumers select this backend at the storage boundary rather than
embedding allocator types in domain algorithms.

## Choosing a backend

Use borrowed storage for a temporary view, stack storage for small fixed
working sets, vector storage for ordinary owned results, and Mnemosyne storage
when the application has selected Atlas allocator policy. Changing storage
does not require a second array algorithm: the same layout and generic
operations are reused for each backend.
