# md5

A high-speed, `no_std` compatible MD5 implementation in pure safe Rust (2024 edition).

## Features

- **Fast** — **4–6% faster** than [RustCrypto `md-5`](https://crates.io/crates/md-5) (~820 MiB/s on Apple Silicon)
- **Optimized** — F/G dependency shortcuts from [animetosho/md5-optimisation](https://github.com/animetosho/md5-optimisation), multi-block batching, zero-cost block casting
- **`no_std`** — works in embedded / restricted environments out of the box
- **Streaming** — incremental `update()` API with method chaining
- **`std` optional** — enables `io::Write` impl via the `std` feature (on by default)

## Usage

```rust
// One-shot
let digest = md5::compute(b"hello world");
assert_eq!(format!("{digest:x}"), "5eb63bbbe01eeed093cb22bb8f5acdc3");

// Incremental
let mut hasher = md5::Md5::new();
hasher.update(b"hello ").update(b"world");
let digest = hasher.finalize();
```

### `no_std`

```toml
[dependencies]
md5 = { version = "0.1", default-features = false }
```

## Benchmarks

Run with `cargo bench`. Comparison against RustCrypto `md-5` on Apple Silicon (M-series):

| Buffer size | Ours (MiB/s) | RustCrypto (MiB/s) | Delta |
| ----------- | ------------ | ------------------ | ----- |
| 64 B        | ~439         | ~414               | +6%   |
| 256 B       | ~682         | ~641               | +6%   |
| 1 KB        | ~782         | ~742               | +5%   |
| 4 KB        | ~812         | ~773               | +5%   |
| 16 KB       | ~819         | ~781               | +5%   |
| 64 KB       | ~815         | ~781               | +4%   |
| 1 MB        | ~811         | ~782               | +4%   |

## License

MIT
