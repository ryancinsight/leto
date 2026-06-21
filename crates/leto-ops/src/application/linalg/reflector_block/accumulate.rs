//! Compact-WY factor construction (Schreiber–Van Loan / LAPACK `dlarft`).

use crate::domain::real::RealScalar;

/// Build the upper-triangular compact-WY factor `T` (`r × r`, row-major) for a
/// panel of `r` Householder reflectors `Hⱼ = I − βⱼ vⱼ vⱼᵀ`, the `vⱼ` stored as
/// the columns of `V` (`m × r`, row-major, leading dimension `r`), so that
/// `H₁ H₂ … H_r = I − V T Vᵀ`.
///
/// The `vⱼ` need not be unit-scaled (leto's QR stores `vⱼ[j] = headⱼ`); the
/// identity holds for any scaling as long as `βⱼ = 2/(vⱼᵀvⱼ)`, because that fixes
/// the reflector `Hⱼ`. Column `j` is built by the Schreiber–Van Loan recurrence
/// `T_{0:j,j} = −βⱼ · T_{0:j,0:j} · (V_{:,0:j}ᵀ vⱼ)`, `T_{jj} = βⱼ`, using the
/// already-built leading block `T_{0:j,0:j}` (upper-triangular).
///
/// Cost `O(m r² + r³)`; `r` is the panel width (small, fixed), so this is a lower
/// order term against the `O(m n r)` trailing GEMM it enables.
pub(super) fn build_t<T: RealScalar>(v: &[T], beta: &[T], t: &mut [T], m: usize, r: usize) {
    t.fill(T::ZERO);
    let mut z_stack = [T::ZERO; 128];
    let mut z_vec = Vec::new();
    for j in 0..r {
        t[j * r + j] = beta[j];
        if j == 0 {
            continue;
        }
        // z = V[:, 0:j]ᵀ · v_j  (length j). Column i of V is zero above row i and
        // column j above row j, so only rows ≥ j contribute (`vrj == 0` skips them).
        let z = if j <= 128 {
            &mut z_stack[..j]
        } else {
            z_vec.resize(j, T::ZERO);
            &mut z_vec[..]
        };
        z.fill(T::ZERO);
        for row in 0..m {
            let vrj = v[row * r + j];
            if vrj == T::ZERO {
                continue;
            }
            for (i, zi) in z.iter_mut().enumerate() {
                *zi = zi.add(v[row * r + i].mul(vrj));
            }
        }
        // T[0:j, j] = −β_j · (T[0:j, 0:j] · z); T is upper-triangular so the inner
        // sum runs l = i..j.
        let neg_beta = T::ZERO.sub(beta[j]);
        for i in 0..j {
            let mut acc = T::ZERO;
            for l in i..j {
                acc = acc.add(t[i * r + l].mul(z[l]));
            }
            t[i * r + j] = neg_beta.mul(acc);
        }
    }
}
