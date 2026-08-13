# leto-python

Thin PyO3/NumPy boundary over the [Leto](../../README.md) array and kernel
crates.

This crate is a binding surface only: it converts NumPy arrays to and from
Leto arrays, maps Leto errors onto Python exceptions, and releases the GIL
around compute. All numerics live in [`leto`](../leto/README.md) and
[`leto-ops`](../leto-ops/README.md).

The extension module is built as `leto_python`. It is distributed as a wheel
and is marked `publish = false` for crates.io.

## Building

```sh
maturin develop -m crates/leto-python/Cargo.toml
```

Wheels for CPython 3.9-3.13 on Linux, Windows, and macOS are built from
GitHub Releases tagged `leto-python-v<version>` and published to PyPI through
OIDC Trusted Publishing.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
