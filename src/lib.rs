//! A high-speed, `no_std` compatible MD5 implementation (RFC 1321).
//!
//! # Examples
//!
//! ```
//! let digest = md5::compute(b"hello world");
//! assert_eq!(
//!     format!("{digest:x}"),
//!     "5eb63bbbe01eeed093cb22bb8f5acdc3"
//! );
//! ```

#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::fmt;

/// Pre-computed round constants: T[i] = floor(2^32 * |sin(i + 1)|).
#[rustfmt::skip]
const T: [u32; 64] = [
    0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
    0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
    0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
    0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
    0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
    0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
    0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
    0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
    0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
    0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
    0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
    0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
    0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
    0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
    0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
    0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
];

/// A 128-bit MD5 digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest(pub [u8; 16]);

impl From<[u8; 16]> for Digest {
    #[inline]
    fn from(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

impl From<Digest> for [u8; 16] {
    #[inline]
    fn from(digest: Digest) -> Self {
        digest.0
    }
}

impl AsRef<[u8; 16]> for Digest {
    #[inline]
    fn as_ref(&self) -> &[u8; 16] {
        &self.0
    }
}

impl AsRef<[u8]> for Digest {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::LowerHex for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::UpperHex for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02X}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(self, f)
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest(\"{self:x}\")")
    }
}

/// Incremental MD5 hasher.
///
/// # Examples
///
/// ```
/// let mut hasher = md5::Md5::new();
/// hasher.update(b"hello ").update(b"world");
/// let digest = hasher.finalize();
/// assert_eq!(format!("{digest:x}"), "5eb63bbbe01eeed093cb22bb8f5acdc3");
/// ```
#[derive(Clone)]
pub struct Md5 {
    state: [u32; 4],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u64,
}

impl Default for Md5 {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Md5 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Md5")
            .field("total_len", &self.total_len)
            .field("buffer_len", &self.buffer_len)
            .finish_non_exhaustive()
    }
}

impl Md5 {
    /// Initial hash values (A, B, C, D) per RFC 1321.
    const INIT: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];

    /// Creates a new MD5 hasher.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Self::INIT,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    /// Feeds data into the hasher.
    ///
    /// Returns `&mut Self` to allow method chaining.
    #[allow(clippy::missing_panics_doc)]
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);

        // If we have buffered data, try to complete a block.
        if self.buffer_len > 0 {
            let need = 64 - self.buffer_len;
            if data.len() < need {
                self.buffer[self.buffer_len..self.buffer_len + data.len()].copy_from_slice(data);
                self.buffer_len += data.len();
                return self;
            }
            self.buffer[self.buffer_len..64].copy_from_slice(&data[..need]);
            let block = self.buffer;
            compress(&mut self.state, &block);
            self.buffer_len = 0;
            data = &data[need..];
        }

        // Process full 64-byte blocks directly from input.
        let n_blocks = data.len() / 64;
        if n_blocks > 0 {
            // SAFETY: data.len() >= n_blocks * 64, and [u8; 64] has the same
            // layout as 64 contiguous u8s with no padding or alignment requirement.
            let blocks: &[[u8; 64]] =
                unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), n_blocks) };
            compress_blocks(&mut self.state, blocks);
            data = &data[n_blocks * 64..];
        }

        // Buffer remaining bytes.
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }

        self
    }

    /// Finalizes the hash and returns the digest.
    #[must_use]
    pub fn finalize(mut self) -> Digest {
        let bit_len = self.total_len.wrapping_mul(8);

        // Append the 0x80 byte.
        self.update(&[0x80]);

        // Pad with zeros until buffer_len ≡ 56 (mod 64).
        if self.buffer_len > 56 {
            let zeros = [0u8; 64];
            self.update(&zeros[..64 - self.buffer_len]);
        }
        if self.buffer_len < 56 {
            let zeros = [0u8; 56];
            let pad = 56 - self.buffer_len;
            self.update(&zeros[..pad]);
        }

        // Append original length in bits as 64-bit LE.
        debug_assert_eq!(self.buffer_len, 56);
        self.buffer[56..64].copy_from_slice(&bit_len.to_le_bytes());
        let block = self.buffer;
        compress(&mut self.state, &block);

        // Produce the final digest in little-endian byte order.
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&self.state[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.state[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.state[2].to_le_bytes());
        out[12..16].copy_from_slice(&self.state[3].to_le_bytes());
        Digest(out)
    }
}

/// Convenience function: compute the MD5 digest of a byte slice.
#[inline]
#[must_use]
pub fn compute(data: &[u8]) -> Digest {
    let mut h = Md5::new();
    h.update(data);
    h.finalize()
}

#[cfg(feature = "std")]
impl std::io::Write for Md5 {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.update(buf);
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[allow(clippy::inline_always, clippy::many_single_char_names)]
#[inline(always)]
fn op_f(w: u32, x: u32, y: u32, z: u32, m: u32, c: u32, s: u32) -> u32 {
    // Dependency shortcut: F(b,c,d) = (b & c) | (~b & d). Since the two AND
    // terms have non-overlapping bits, OR can be replaced with ADD. This lets
    // (~b & d) feed into the accumulator without waiting for the OR gate.
    (!x & z)
        .wrapping_add(w)
        .wrapping_add(m)
        .wrapping_add(c)
        .wrapping_add(x & y)
        .rotate_left(s)
        .wrapping_add(x)
}

#[allow(clippy::inline_always, clippy::many_single_char_names)]
#[inline(always)]
fn op_g(w: u32, x: u32, y: u32, z: u32, m: u32, c: u32, s: u32) -> u32 {
    // Dependency shortcut: since (x & z) and (y & !z) have non-overlapping bits,
    // OR can be replaced with ADD. This lets us delay the dependency on x (= b)
    // by one operation — x only needs AND before joining the addition chain.
    // See: https://github.com/animetosho/md5-optimisation
    (y & !z)
        .wrapping_add(w)
        .wrapping_add(m)
        .wrapping_add(c)
        .wrapping_add(x & z)
        .rotate_left(s)
        .wrapping_add(x)
}

#[allow(clippy::inline_always, clippy::many_single_char_names)]
#[inline(always)]
fn op_h(w: u32, x: u32, y: u32, z: u32, m: u32, c: u32, s: u32) -> u32 {
    (x ^ y ^ z)
        .wrapping_add(w)
        .wrapping_add(m)
        .wrapping_add(c)
        .rotate_left(s)
        .wrapping_add(x)
}

#[allow(clippy::inline_always, clippy::many_single_char_names)]
#[inline(always)]
fn op_i(w: u32, x: u32, y: u32, z: u32, m: u32, c: u32, s: u32) -> u32 {
    (y ^ (x | !z))
        .wrapping_add(w)
        .wrapping_add(m)
        .wrapping_add(c)
        .rotate_left(s)
        .wrapping_add(x)
}

/// Compress multiple 64-byte blocks into the state.
#[inline]
fn compress_blocks(state: &mut [u32; 4], blocks: &[[u8; 64]]) {
    for block in blocks {
        compress(state, block);
    }
}

/// Compress a single 64-byte block into the state.
///
/// This is the performance-critical inner loop. We fully unroll all 64 rounds
/// and use `wrapping_add` / `rotate_left` which map to efficient instructions.
#[allow(clippy::many_single_char_names)]
#[inline]
fn compress(state: &mut [u32; 4], block: &[u8; 64]) {
    // Decode block into sixteen 32-bit little-endian words.
    let mut m = [0u32; 16];
    for (o, chunk) in m.iter_mut().zip(block.chunks_exact(4)) {
        *o = u32::from_le_bytes(chunk.try_into().unwrap());
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];

    // Round 1 (F)
    a = op_f(a, b, c, d, m[0], T[0], 7);
    d = op_f(d, a, b, c, m[1], T[1], 12);
    c = op_f(c, d, a, b, m[2], T[2], 17);
    b = op_f(b, c, d, a, m[3], T[3], 22);
    a = op_f(a, b, c, d, m[4], T[4], 7);
    d = op_f(d, a, b, c, m[5], T[5], 12);
    c = op_f(c, d, a, b, m[6], T[6], 17);
    b = op_f(b, c, d, a, m[7], T[7], 22);
    a = op_f(a, b, c, d, m[8], T[8], 7);
    d = op_f(d, a, b, c, m[9], T[9], 12);
    c = op_f(c, d, a, b, m[10], T[10], 17);
    b = op_f(b, c, d, a, m[11], T[11], 22);
    a = op_f(a, b, c, d, m[12], T[12], 7);
    d = op_f(d, a, b, c, m[13], T[13], 12);
    c = op_f(c, d, a, b, m[14], T[14], 17);
    b = op_f(b, c, d, a, m[15], T[15], 22);

    // Round 2 (G)
    a = op_g(a, b, c, d, m[1], T[16], 5);
    d = op_g(d, a, b, c, m[6], T[17], 9);
    c = op_g(c, d, a, b, m[11], T[18], 14);
    b = op_g(b, c, d, a, m[0], T[19], 20);
    a = op_g(a, b, c, d, m[5], T[20], 5);
    d = op_g(d, a, b, c, m[10], T[21], 9);
    c = op_g(c, d, a, b, m[15], T[22], 14);
    b = op_g(b, c, d, a, m[4], T[23], 20);
    a = op_g(a, b, c, d, m[9], T[24], 5);
    d = op_g(d, a, b, c, m[14], T[25], 9);
    c = op_g(c, d, a, b, m[3], T[26], 14);
    b = op_g(b, c, d, a, m[8], T[27], 20);
    a = op_g(a, b, c, d, m[13], T[28], 5);
    d = op_g(d, a, b, c, m[2], T[29], 9);
    c = op_g(c, d, a, b, m[7], T[30], 14);
    b = op_g(b, c, d, a, m[12], T[31], 20);

    // Round 3 (H)
    a = op_h(a, b, c, d, m[5], T[32], 4);
    d = op_h(d, a, b, c, m[8], T[33], 11);
    c = op_h(c, d, a, b, m[11], T[34], 16);
    b = op_h(b, c, d, a, m[14], T[35], 23);
    a = op_h(a, b, c, d, m[1], T[36], 4);
    d = op_h(d, a, b, c, m[4], T[37], 11);
    c = op_h(c, d, a, b, m[7], T[38], 16);
    b = op_h(b, c, d, a, m[10], T[39], 23);
    a = op_h(a, b, c, d, m[13], T[40], 4);
    d = op_h(d, a, b, c, m[0], T[41], 11);
    c = op_h(c, d, a, b, m[3], T[42], 16);
    b = op_h(b, c, d, a, m[6], T[43], 23);
    a = op_h(a, b, c, d, m[9], T[44], 4);
    d = op_h(d, a, b, c, m[12], T[45], 11);
    c = op_h(c, d, a, b, m[15], T[46], 16);
    b = op_h(b, c, d, a, m[2], T[47], 23);

    // Round 4 (I)
    a = op_i(a, b, c, d, m[0], T[48], 6);
    d = op_i(d, a, b, c, m[7], T[49], 10);
    c = op_i(c, d, a, b, m[14], T[50], 15);
    b = op_i(b, c, d, a, m[5], T[51], 21);
    a = op_i(a, b, c, d, m[12], T[52], 6);
    d = op_i(d, a, b, c, m[3], T[53], 10);
    c = op_i(c, d, a, b, m[10], T[54], 15);
    b = op_i(b, c, d, a, m[1], T[55], 21);
    a = op_i(a, b, c, d, m[8], T[56], 6);
    d = op_i(d, a, b, c, m[15], T[57], 10);
    c = op_i(c, d, a, b, m[6], T[58], 15);
    b = op_i(b, c, d, a, m[13], T[59], 21);
    a = op_i(a, b, c, d, m[4], T[60], 6);
    d = op_i(d, a, b, c, m[11], T[61], 10);
    c = op_i(c, d, a, b, m[2], T[62], 15);
    b = op_i(b, c, d, a, m[9], T[63], 21);

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::format;

    /// RFC 1321 test vectors.
    #[test]
    fn rfc1321_test_vectors() {
        let cases: &[(&[u8], &str)] = &[
            (b"", "d41d8cd98f00b204e9800998ecf8427e"),
            (b"a", "0cc175b9c0f1b6a831c399e269772661"),
            (b"abc", "900150983cd24fb0d6963f7d28e17f72"),
            (b"message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                b"abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                b"12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];

        for &(input, expected) in cases {
            let digest = compute(input);
            let hex = format!("{digest:x}");
            assert_eq!(
                hex,
                expected,
                "failed for input: {:?}",
                core::str::from_utf8(input)
            );
        }
    }

    #[test]
    fn incremental_matches_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = compute(data);

        let mut hasher = Md5::new();
        for byte in data {
            hasher.update(core::slice::from_ref(byte));
        }
        let incremental = hasher.finalize();

        assert_eq!(oneshot, incremental);
    }

    #[test]
    fn display_and_debug() {
        let d = compute(b"");
        assert_eq!(format!("{d}"), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(format!("{d:X}"), "D41D8CD98F00B204E9800998ECF8427E");
        assert_eq!(
            format!("{d:?}"),
            "Digest(\"d41d8cd98f00b204e9800998ecf8427e\")"
        );
    }

    #[test]
    fn hello_world() {
        let d = compute(b"hello world");
        assert_eq!(format!("{d:x}"), "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    fn cross_block_boundary() {
        // 55 bytes — fits in one block with padding.
        let d55 = compute(&[0xAA; 55]);
        // 56 bytes — padding spills into a second block.
        let d56 = compute(&[0xAA; 56]);
        // 64 bytes — exactly one data block + one padding block.
        let d64 = compute(&[0xAA; 64]);
        // 128 bytes — two full data blocks + padding block.
        let d128 = compute(&[0xAA; 128]);

        // Verify they're all different and don't panic.
        assert_ne!(d55, d56);
        assert_ne!(d56, d64);
        assert_ne!(d64, d128);
    }

    #[test]
    fn large_input() {
        // 1 MB of zeros
        let data = std::vec![0u8; 1024 * 1024];
        let d = compute(&data);
        assert_eq!(format!("{d:x}"), "b6d81b360a5672d80c27430f39153e2c");
    }

    #[test]
    fn chaining() {
        let mut hasher = Md5::new();
        hasher.update(b"hello ").update(b"world");
        let d = hasher.finalize();
        assert_eq!(format!("{d:x}"), "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    fn from_into_conversions() {
        let d = compute(b"abc");
        let bytes: [u8; 16] = d.into();
        assert_eq!(bytes, d.0);

        let d2 = Digest::from(bytes);
        assert_eq!(d, d2);
    }

    #[test]
    fn as_ref_slice() {
        let d = compute(b"abc");
        let slice: &[u8] = d.as_ref();
        assert_eq!(slice.len(), 16);
        assert_eq!(slice, &d.0[..]);
    }

    #[test]
    fn md5_debug() {
        let hasher = Md5::new();
        let dbg = format!("{hasher:?}");
        assert!(dbg.contains("Md5"));
        assert!(dbg.contains("total_len"));
    }

    #[test]
    fn io_write() {
        use std::io::Write;
        let mut hasher = Md5::new();
        hasher.write_all(b"hello world").unwrap();
        let d = hasher.finalize();
        assert_eq!(format!("{d:x}"), "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }
}
