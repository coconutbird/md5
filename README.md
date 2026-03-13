# md5

A high-speed, `no_std` compatible MD5 implementation in pure safe Rust (2024 edition).

## Features

- **Fast** — matches [RustCrypto `md-5`](https://crates.io/crates/md-5) throughput (~800 MiB/s)
- **Safe** — `#![forbid(unsafe_code)]`, zero `unsafe` blocks
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

Run with `cargo bench`. Comparison against RustCrypto `md-5` on Apple Silicon:

| Buffer size | Ours (MiB/s) | RustCrypto (MiB/s) |
| ----------- | ------------ | ------------------ |
| 64 B        | ~416         | ~425               |
| 256 B       | ~637         | ~653               |
| 1 KB        | ~750         | ~753               |
| 4 KB        | ~795         | ~785               |
| 16 KB       | ~794         | ~785               |
| 64 KB       | ~800         | ~807               |
| 1 MB        | ~794         | ~790               |

## License

MIT
