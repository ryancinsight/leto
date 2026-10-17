//! Public quaternion inverse bits for the independent exact-rational oracle.
use eunomia::RealField;
use leto::geometry::Quaternion;
use std::io::{self, BufRead, Write};
fn observe<T: RealField>(
    output: &mut impl Write,
    index: usize,
    v: [T; 4],
    bits: impl Fn(T) -> u64,
) -> io::Result<()> {
    let q = Quaternion::new(v[0], v[1], v[2], v[3]);
    let norm = q.norm_squared();
    if let Some(r) = q.try_inverse() {
        let left = q * r;
        let right = r * q;
        writeln!(
            output,
            "{index} {} Some {} {} {} {} {} {} {} {} {} {} {} {}",
            bits(norm),
            bits(r.w),
            bits(r.x),
            bits(r.y),
            bits(r.z),
            bits(left.w),
            bits(left.x),
            bits(left.y),
            bits(left.z),
            bits(right.w),
            bits(right.x),
            bits(right.y),
            bits(right.z)
        )
    } else {
        writeln!(output, "{index} {} None", bits(norm))
    }
}
fn main() -> io::Result<()> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line = line.expect("fixture");
        let mut fields = line.split_whitespace();
        let width = fields.next().expect("precision");
        let v: Vec<_> = fields
            .map(|s| u64::from_str_radix(s, 16).expect("IEEE bits"))
            .collect();
        assert_eq!(v.len(), 4);
        match width {
            "32" => observe(
                &mut output,
                index,
                std::array::from_fn(|i| f32::from_bits(u32::try_from(v[i]).expect("binary32"))),
                |x: f32| u64::from(x.to_bits()),
            ),
            "64" => observe(
                &mut output,
                index,
                std::array::from_fn(|i| f64::from_bits(v[i])),
                f64::to_bits,
            ),
            _ => panic!("precision"),
        }?;
    }
    output.flush()
}
