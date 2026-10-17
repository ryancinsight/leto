//! Public normalization/checked-unit output bits for an external oracle.
use eunomia::RealField;
use leto::geometry::{Unit, Vector};
use std::io::{self, BufRead, Write};

fn observe<T: RealField, const N: usize>(
    output: &mut impl Write,
    index: usize,
    kind: &str,
    values: &[u64],
    decode: impl Fn(u64) -> T,
    bits: impl Fn(T) -> u64,
) -> io::Result<()> {
    let v: Vector<T, N> = Vector::from_array(std::array::from_fn(|i| decode(values[i])));
    write!(output, "{index} {kind} {}", bits(v.norm()))?;
    for x in v.normalize().data {
        write!(output, " {}", bits(x))?;
    }
    if let Some(unit) = Unit::try_new(v, T::ZERO) {
        write!(output, " Some")?;
        for x in unit.into_inner().data {
            write!(output, " {}", bits(x))?;
        }
    } else {
        write!(output, " None")?;
    }
    writeln!(output)
}

fn dimension<const N: usize>(output: &mut impl Write, index: usize, width: u32, v: &[u64]) -> io::Result<()> {
    assert_eq!(v.len(), N);
    if width == 32 {
        observe::<f32, N>(output,index,"native",v,|b|f32::from_bits(b as u32),|v|u64::from(v.to_bits()))?;
        observe::<eunomia::F32, N>(output,index,"wrapped",v,|b|eunomia::F32(f32::from_bits(b as u32)),|v|u64::from(v.0.to_bits()))?;
    } else {
        assert_eq!(width, 64);
        observe::<f64, N>(output,index,"native",v,f64::from_bits,f64::to_bits)?;
        observe::<eunomia::F64, N>(output,index,"wrapped",v,|b|eunomia::F64(f64::from_bits(b)),|v|v.0.to_bits())?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for (index, line) in io::stdin().lock().lines().enumerate() {
        let line = line?;
        let mut fields = line.split_whitespace();
        let width = fields.next().expect("width").parse().expect("precision");
        let n = fields.next().expect("dimension").parse().expect("dimension");
        let v: Vec<_> = fields.map(|s|u64::from_str_radix(s,16).expect("IEEE bits")).collect();
        match n {
            0 => dimension::<0>(&mut output,index,width,&v),
            1 => dimension::<1>(&mut output,index,width,&v),
            2 => dimension::<2>(&mut output,index,width,&v),
            3 => dimension::<3>(&mut output,index,width,&v),
            4 => dimension::<4>(&mut output,index,width,&v),
            8 => dimension::<8>(&mut output,index,width,&v),
            17 => dimension::<17>(&mut output,index,width,&v),
            _ => panic!("fixture dimension"),
        }?;
    }
    output.flush()
}
