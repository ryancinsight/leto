//! Checked construction of rigid rotations from orthonormal world-space bases.

use super::{Quaternion, UnitQuaternion, Vector3};
use eunomia::RealField;
use thiserror::Error;

/// Why a proposed set of rotation-matrix columns cannot define a rigid rotation.
///
/// [`UnitQuaternion::try_from_rotation_columns`] accepts only finite,
/// right-handed, orthonormal columns. Keeping that validation at the geometry
/// boundary prevents an affine scale, reflection, or malformed external input
/// from being represented as an [`Isometry3`](super::Isometry3).
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum RotationBasisError {
    /// The caller supplied a tolerance that cannot define a meaningful unit
    /// basis acceptance region.
    #[error("rotation-basis tolerance must be finite and lie in (0, 1)")]
    InvalidTolerance,

    /// One of the proposed axes contains a non-finite component.
    #[error("rotation-basis {axis}-axis contains a non-finite component")]
    NonFiniteAxis {
        /// The invalid basis axis.
        axis: &'static str,
    },

    /// One of the proposed axes is not unit length within the supplied
    /// tolerance.
    #[error("rotation-basis {axis}-axis is not unit length within tolerance")]
    NonUnitAxis {
        /// The invalid basis axis.
        axis: &'static str,
    },

    /// Two proposed axes are not orthogonal within the supplied tolerance.
    #[error("rotation-basis {first}- and {second}-axes are not orthogonal within tolerance")]
    NonOrthogonal {
        /// The first non-orthogonal axis.
        first: &'static str,
        /// The second non-orthogonal axis.
        second: &'static str,
    },

    /// The proposed basis is not right-handed.
    #[error("rotation-basis columns do not form a right-handed frame")]
    NotRightHanded,
}

impl<T: RealField> UnitQuaternion<T> {
    /// Construct a rotation from its world-space matrix columns.
    ///
    /// The supplied `x_axis`, `y_axis`, and `z_axis` are the columns of the
    /// column-vector rotation matrix. They must be finite, unit length,
    /// pairwise orthogonal, and right-handed within `tolerance`. The method
    /// does not orthonormalize or otherwise repair the columns: an affine basis
    /// outside that declared acceptance region is rejected rather than silently
    /// changing its physical meaning. The returned quaternion is normalized so an
    /// accepted, rounded representation still satisfies the unit-rotation type
    /// invariant.
    ///
    /// The returned unit quaternion maps local basis vectors to the supplied
    /// world-space axes. For example, columns `(0,1,0)`, `(-1,0,0)`, and
    /// `(0,0,1)` rotate the local positive x-axis onto world positive y.
    ///
    /// # Errors
    /// Returns [`RotationBasisError`] when `tolerance` is invalid, a component
    /// is non-finite, the columns are not orthonormal, or the frame is
    /// left-handed.
    #[inline]
    pub fn try_from_rotation_columns(
        x_axis: Vector3<T>,
        y_axis: Vector3<T>,
        z_axis: Vector3<T>,
        tolerance: T,
    ) -> core::result::Result<Self, RotationBasisError> {
        let zero = T::ZERO;
        let one = T::ONE;
        if !tolerance.is_finite() || tolerance <= zero || tolerance >= one {
            return Err(RotationBasisError::InvalidTolerance);
        }

        for (axis_name, axis) in [("x", x_axis), ("y", y_axis), ("z", z_axis)] {
            if axis.data.iter().any(|component| !component.is_finite()) {
                return Err(RotationBasisError::NonFiniteAxis { axis: axis_name });
            }
            if (axis.norm() - one).abs() > tolerance {
                return Err(RotationBasisError::NonUnitAxis { axis: axis_name });
            }
        }

        for (first, second, first_name, second_name) in [
            (x_axis, y_axis, "x", "y"),
            (x_axis, z_axis, "x", "z"),
            (y_axis, z_axis, "y", "z"),
        ] {
            if first.dot(second).abs() > tolerance {
                return Err(RotationBasisError::NonOrthogonal {
                    first: first_name,
                    second: second_name,
                });
            }
        }

        if (x_axis.cross(y_axis).dot(z_axis) - one).abs() > tolerance {
            return Err(RotationBasisError::NotRightHanded);
        }

        let [m00, m10, m20] = x_axis.data;
        let [m01, m11, m21] = y_axis.data;
        let [m02, m12, m22] = z_axis.data;
        let two = one + one;
        let four = two + two;
        let trace = m00 + m11 + m22;

        let quaternion = if trace > zero {
            let scale = (trace + one).sqrt() * two;
            Quaternion::new(
                scale / four,
                (m21 - m12) / scale,
                (m02 - m20) / scale,
                (m10 - m01) / scale,
            )
        } else if m00 > m11 && m00 > m22 {
            let scale = (one + m00 - m11 - m22).sqrt() * two;
            Quaternion::new(
                (m21 - m12) / scale,
                scale / four,
                (m01 + m10) / scale,
                (m02 + m20) / scale,
            )
        } else if m11 > m22 {
            let scale = (one + m11 - m00 - m22).sqrt() * two;
            Quaternion::new(
                (m02 - m20) / scale,
                (m01 + m10) / scale,
                scale / four,
                (m12 + m21) / scale,
            )
        } else {
            let scale = (one + m22 - m00 - m11).sqrt() * two;
            Quaternion::new(
                (m10 - m01) / scale,
                (m02 + m20) / scale,
                (m12 + m21) / scale,
                scale / four,
            )
        };

        // An accepted basis is within the caller's declared rotation tolerance,
        // so this finite quaternion has non-zero norm. Normalization makes the
        // representation exact at the unit-quaternion boundary without
        // repairing the input columns themselves.
        Ok(Self::new_unchecked(quaternion.normalize()))
    }
}
