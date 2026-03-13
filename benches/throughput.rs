use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rustcrypto_md5::Digest as RustCryptoDigest;
use rustcrypto_md5::Md5 as RustCryptoMd5;

const SIZES: &[usize] = &[64, 256, 1024, 4096, 16384, 65536, 1_048_576];

fn bench_ours(c: &mut Criterion) {
    let mut group = c.benchmark_group("ours");
    for &size in SIZES {
        let data = vec![0xABu8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| md5::compute(std::hint::black_box(data)));
        });
    }
    group.finish();
}

fn bench_rustcrypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("md-5 (RustCrypto)");
    for &size in SIZES {
        let data = vec![0xABu8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| RustCryptoMd5::digest(std::hint::black_box(data)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ours, bench_rustcrypto);
criterion_main!(benches);
