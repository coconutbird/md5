//! NEON multi-buffer MD5: hash 4 independent inputs in parallel using
//! `AArch64` NEON 128-bit registers (4×u32 lanes).
#![allow(
    clippy::inline_always,
    clippy::many_single_char_names,
    clippy::wildcard_imports
)]

use core::arch::aarch64::*;

use crate::T;

/// F(b,c,d) = bitwise select: for each bit, pick c where b=1, d where b=0.
#[inline(always)]
unsafe fn f(b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint32x4_t {
    unsafe { vbslq_u32(b, c, d) }
}

/// G(b,c,d) = bitwise select with d as mask.
#[inline(always)]
unsafe fn g(b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint32x4_t {
    unsafe { vbslq_u32(d, b, c) }
}

/// H(b,c,d) = b ^ c ^ d.
#[inline(always)]
unsafe fn h(b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint32x4_t {
    unsafe { veorq_u32(veorq_u32(b, c), d) }
}

/// I(b,c,d) = c ^ (b | ~d).
#[inline(always)]
unsafe fn i(b: uint32x4_t, c: uint32x4_t, d: uint32x4_t) -> uint32x4_t {
    unsafe { veorq_u32(c, vorrq_u32(b, vmvnq_u32(d))) }
}

/// One MD5 round step: a = ((a + func + m + T[k]) <<< s) + b.
/// Uses a macro because `rotate_left` needs a literal and const generics
/// can't do `32 - N` in expressions.
macro_rules! round {
    ($a:expr, $b:expr, $func:expr, $m:expr, $tc:expr, $s:literal) => {
        unsafe {
            let t = vaddq_u32($a, $func);
            let t = vaddq_u32(t, $m);
            let t = vaddq_u32(t, vdupq_n_u32($tc));
            let t = vorrq_u32(vshlq_n_u32(t, $s), vshrq_n_u32(t, 32 - $s));
            vaddq_u32(t, $b)
        }
    };
}

/// Load message word `idx` from 4 blocks into a NEON register (one lane per input).
#[inline(always)]
unsafe fn load_word(blocks: &[&[u8; 64]; 4], idx: usize) -> uint32x4_t {
    unsafe {
        let off = idx * 4;
        let w0 = u32::from_le_bytes([
            blocks[0][off],
            blocks[0][off + 1],
            blocks[0][off + 2],
            blocks[0][off + 3],
        ]);
        let w1 = u32::from_le_bytes([
            blocks[1][off],
            blocks[1][off + 1],
            blocks[1][off + 2],
            blocks[1][off + 3],
        ]);
        let w2 = u32::from_le_bytes([
            blocks[2][off],
            blocks[2][off + 1],
            blocks[2][off + 2],
            blocks[2][off + 3],
        ]);
        let w3 = u32::from_le_bytes([
            blocks[3][off],
            blocks[3][off + 1],
            blocks[3][off + 2],
            blocks[3][off + 3],
        ]);
        vld1q_u32([w0, w1, w2, w3].as_ptr())
    }
}

/// Compress one 64-byte block for 4 independent hash states in parallel.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn compress4(states: &mut [uint32x4_t; 4], blocks: &[&[u8; 64]; 4]) {
    // Load 16 message words (each register = word i from all 4 inputs).
    let m: [uint32x4_t; 16] = unsafe {
        let mut m = [vdupq_n_u32(0); 16];
        let mut idx = 0;
        while idx < 16 {
            m[idx] = load_word(blocks, idx);
            idx += 1;
        }
        m
    };

    let (mut a, mut b, mut c, mut d) = (states[0], states[1], states[2], states[3]);

    // Round 1 (F)
    a = round!(a, b, f(b, c, d), m[0], T[0], 7);
    d = round!(d, a, f(a, b, c), m[1], T[1], 12);
    c = round!(c, d, f(d, a, b), m[2], T[2], 17);
    b = round!(b, c, f(c, d, a), m[3], T[3], 22);
    a = round!(a, b, f(b, c, d), m[4], T[4], 7);
    d = round!(d, a, f(a, b, c), m[5], T[5], 12);
    c = round!(c, d, f(d, a, b), m[6], T[6], 17);
    b = round!(b, c, f(c, d, a), m[7], T[7], 22);
    a = round!(a, b, f(b, c, d), m[8], T[8], 7);
    d = round!(d, a, f(a, b, c), m[9], T[9], 12);
    c = round!(c, d, f(d, a, b), m[10], T[10], 17);
    b = round!(b, c, f(c, d, a), m[11], T[11], 22);
    a = round!(a, b, f(b, c, d), m[12], T[12], 7);
    d = round!(d, a, f(a, b, c), m[13], T[13], 12);
    c = round!(c, d, f(d, a, b), m[14], T[14], 17);
    b = round!(b, c, f(c, d, a), m[15], T[15], 22);

    // Round 2 (G)
    a = round!(a, b, g(b, c, d), m[1], T[16], 5);
    d = round!(d, a, g(a, b, c), m[6], T[17], 9);
    c = round!(c, d, g(d, a, b), m[11], T[18], 14);
    b = round!(b, c, g(c, d, a), m[0], T[19], 20);
    a = round!(a, b, g(b, c, d), m[5], T[20], 5);
    d = round!(d, a, g(a, b, c), m[10], T[21], 9);
    c = round!(c, d, g(d, a, b), m[15], T[22], 14);
    b = round!(b, c, g(c, d, a), m[4], T[23], 20);
    a = round!(a, b, g(b, c, d), m[9], T[24], 5);
    d = round!(d, a, g(a, b, c), m[14], T[25], 9);
    c = round!(c, d, g(d, a, b), m[3], T[26], 14);
    b = round!(b, c, g(c, d, a), m[8], T[27], 20);
    a = round!(a, b, g(b, c, d), m[13], T[28], 5);
    d = round!(d, a, g(a, b, c), m[2], T[29], 9);
    c = round!(c, d, g(d, a, b), m[7], T[30], 14);
    b = round!(b, c, g(c, d, a), m[12], T[31], 20);

    // Round 3 (H)
    a = round!(a, b, h(b, c, d), m[5], T[32], 4);
    d = round!(d, a, h(a, b, c), m[8], T[33], 11);
    c = round!(c, d, h(d, a, b), m[11], T[34], 16);
    b = round!(b, c, h(c, d, a), m[14], T[35], 23);
    a = round!(a, b, h(b, c, d), m[1], T[36], 4);
    d = round!(d, a, h(a, b, c), m[4], T[37], 11);
    c = round!(c, d, h(d, a, b), m[7], T[38], 16);
    b = round!(b, c, h(c, d, a), m[10], T[39], 23);
    a = round!(a, b, h(b, c, d), m[13], T[40], 4);
    d = round!(d, a, h(a, b, c), m[0], T[41], 11);
    c = round!(c, d, h(d, a, b), m[3], T[42], 16);
    b = round!(b, c, h(c, d, a), m[6], T[43], 23);
    a = round!(a, b, h(b, c, d), m[9], T[44], 4);
    d = round!(d, a, h(a, b, c), m[12], T[45], 11);
    c = round!(c, d, h(d, a, b), m[15], T[46], 16);
    b = round!(b, c, h(c, d, a), m[2], T[47], 23);

    // Round 4 (I)
    a = round!(a, b, i(b, c, d), m[0], T[48], 6);
    d = round!(d, a, i(a, b, c), m[7], T[49], 10);
    c = round!(c, d, i(d, a, b), m[14], T[50], 15);
    b = round!(b, c, i(c, d, a), m[5], T[51], 21);
    a = round!(a, b, i(b, c, d), m[12], T[52], 6);
    d = round!(d, a, i(a, b, c), m[3], T[53], 10);
    c = round!(c, d, i(d, a, b), m[10], T[54], 15);
    b = round!(b, c, i(c, d, a), m[1], T[55], 21);
    a = round!(a, b, i(b, c, d), m[8], T[56], 6);
    d = round!(d, a, i(a, b, c), m[15], T[57], 10);
    c = round!(c, d, i(d, a, b), m[6], T[58], 15);
    b = round!(b, c, i(c, d, a), m[13], T[59], 21);
    a = round!(a, b, i(b, c, d), m[4], T[60], 6);
    d = round!(d, a, i(a, b, c), m[11], T[61], 10);
    c = round!(c, d, i(d, a, b), m[2], T[62], 15);
    b = round!(b, c, i(c, d, a), m[9], T[63], 21);

    // Accumulate.
    states[0] = vaddq_u32(states[0], a);
    states[1] = vaddq_u32(states[1], b);
    states[2] = vaddq_u32(states[2], c);
    states[3] = vaddq_u32(states[3], d);
}
