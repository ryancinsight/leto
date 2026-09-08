use eunomia::{layout::cast_slice, Bf16, Pod, F16};
use hermes_simd::LaneScalar;
use leto::Complex;

pub(crate) trait PayloadScalar: LaneScalar + Pod {
    const PAYLOADS: &'static [Self];

    fn ordinal(index: usize) -> Self;
}

impl PayloadScalar for f32 {
    const PAYLOADS: &'static [Self] = &[
        Self::from_bits(0),
        Self::from_bits(0x8000_0000),
        Self::from_bits(0x7f80_0000),
        Self::from_bits(0xff80_0000),
        Self::from_bits(0x7fc0_0001),
        Self::from_bits(0x7fc0_0017),
        Self::from_bits(0xffc0_0031),
        Self::from_bits(0x7f80_0001),
        Self::from_bits(0x7f80_0017),
        Self::from_bits(0xff80_0031),
        Self::from_bits(1),
        Self::from_bits(0x8000_0001),
        Self::from_bits(0x007f_ffff),
        Self::from_bits(0x807f_ffff),
        Self::from_bits(0x3f80_0000),
        Self::from_bits(0xbf80_0000),
    ];

    fn ordinal(index: usize) -> Self {
        Self::from_bits(u32::try_from(index).expect("fixture index fits u32"))
    }
}

impl PayloadScalar for f64 {
    const PAYLOADS: &'static [Self] = &[
        Self::from_bits(0),
        Self::from_bits(0x8000_0000_0000_0000),
        Self::from_bits(0x7ff0_0000_0000_0000),
        Self::from_bits(0xfff0_0000_0000_0000),
        Self::from_bits(0x7ff8_0000_0000_0001),
        Self::from_bits(0x7ff8_0000_0000_0017),
        Self::from_bits(0xfff8_0000_0000_0031),
        Self::from_bits(0x7ff0_0000_0000_0001),
        Self::from_bits(0x7ff0_0000_0000_0017),
        Self::from_bits(0xfff0_0000_0000_0031),
        Self::from_bits(1),
        Self::from_bits(0x8000_0000_0000_0001),
        Self::from_bits(0x000f_ffff_ffff_ffff),
        Self::from_bits(0x800f_ffff_ffff_ffff),
        Self::from_bits(0x3ff0_0000_0000_0000),
        Self::from_bits(0xbff0_0000_0000_0000),
    ];

    fn ordinal(index: usize) -> Self {
        Self::from_bits(u64::try_from(index).expect("fixture index fits u64"))
    }
}

impl PayloadScalar for F16 {
    const PAYLOADS: &'static [Self] = &[
        Self::from_bits(0),
        Self::from_bits(0x8000),
        Self::from_bits(0x7c00),
        Self::from_bits(0xfc00),
        Self::from_bits(0x7e01),
        Self::from_bits(0x7e17),
        Self::from_bits(0xfe31),
        Self::from_bits(0x7c01),
        Self::from_bits(0x7c17),
        Self::from_bits(0xfc31),
        Self::from_bits(1),
        Self::from_bits(0x8001),
        Self::from_bits(0x03ff),
        Self::from_bits(0x83ff),
        Self::from_bits(0x3c00),
        Self::from_bits(0xbc00),
    ];

    fn ordinal(index: usize) -> Self {
        Self::from_bits(u16::try_from(index % 65_536).expect("fixture residue fits u16"))
    }
}

impl PayloadScalar for Bf16 {
    const PAYLOADS: &'static [Self] = &[
        Self::from_bits(0),
        Self::from_bits(0x8000),
        Self::from_bits(0x7f80),
        Self::from_bits(0xff80),
        Self::from_bits(0x7fc1),
        Self::from_bits(0x7fd7),
        Self::from_bits(0xfff1),
        Self::from_bits(0x7f81),
        Self::from_bits(0x7f97),
        Self::from_bits(0xffb1),
        Self::from_bits(1),
        Self::from_bits(0x8001),
        Self::from_bits(0x007f),
        Self::from_bits(0x807f),
        Self::from_bits(0x3f80),
        Self::from_bits(0xbf80),
    ];

    fn ordinal(index: usize) -> Self {
        Self::from_bits(u16::try_from(index % 65_536).expect("fixture residue fits u16"))
    }
}

pub(crate) fn values<T: PayloadScalar>(len: usize) -> Vec<Complex<T>> {
    (0..len)
        .map(|index| {
            // The upper index bits distinguish cycles of the 16-bit ordinal.
            let special = (index + index / 65_536) % T::PAYLOADS.len();
            Complex::new(T::PAYLOADS[special], T::ordinal(index))
        })
        .collect()
}

pub(crate) fn expected<T: Copy>(
    source: &[Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Vec<Complex<T>> {
    let matrix_len = rows * columns;
    let mut output = source.to_vec();
    for matrix in 0..matrix_count {
        let base = matrix * matrix_len;
        for row in 0..rows {
            for column in 0..columns {
                output[base + column * rows + row] = source[base + row * columns + column];
            }
        }
    }
    output
}

pub(crate) fn assert_bits<T: Pod>(actual: &[Complex<T>], expected: &[Complex<T>]) {
    assert_eq!(cast_slice::<_, u8>(actual), cast_slice::<_, u8>(expected));
}
