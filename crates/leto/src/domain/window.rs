//! Validated spatial sliding-window parameters.

use super::error::{LetoError, Result};

/// Validated stride, padding, dilation, and kernel extents for a spatial window.
///
/// The same geometry is used by pooling and unfold/fold operations. Keeping it
/// in Leto makes the output-shape contract independent of any execution
/// backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowParameters<const D: usize> {
    kernel: [usize; D],
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
}

impl<const D: usize> WindowParameters<D> {
    /// Construct validated spatial window parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when a kernel, stride, or dilation
    /// extent is zero.
    pub fn new(
        kernel: [usize; D],
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<Self> {
        if kernel.contains(&0) {
            return Err(LetoError::InvalidInput(
                "window kernel extents must be nonzero".to_string(),
            ));
        }
        if stride.contains(&0) {
            return Err(LetoError::InvalidInput(
                "window stride extents must be nonzero".to_string(),
            ));
        }
        if dilation.contains(&0) {
            return Err(LetoError::InvalidInput(
                "window dilation extents must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            kernel,
            stride,
            padding,
            dilation,
        })
    }

    /// Return the per-axis kernel extents.
    #[must_use]
    pub const fn kernel(&self) -> &[usize; D] {
        &self.kernel
    }

    /// Return the per-axis traversal strides.
    #[must_use]
    pub const fn stride(&self) -> &[usize; D] {
        &self.stride
    }

    /// Return the per-axis symmetric padding.
    #[must_use]
    pub const fn padding(&self) -> &[usize; D] {
        &self.padding
    }

    /// Return the per-axis kernel dilation.
    #[must_use]
    pub const fn dilation(&self) -> &[usize; D] {
        &self.dilation
    }

    /// Return the effective kernel extent on one spatial axis.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when the dilation arithmetic exceeds
    /// `usize`.
    pub fn effective_kernel_extent(&self, axis: usize) -> Result<usize> {
        let kernel = self.kernel.get(axis).ok_or_else(|| {
            LetoError::InvalidInput(format!("window axis {axis} exceeds rank {D}"))
        })?;
        let dilation = self.dilation.get(axis).ok_or_else(|| {
            LetoError::InvalidInput(format!("window axis {axis} exceeds rank {D}"))
        })?;
        dilation
            .checked_mul(kernel - 1)
            .and_then(|extent| extent.checked_add(1))
            .ok_or(LetoError::Overflow {
                reason: "window effective kernel extent",
            })
    }

    /// Derive one output spatial extent from one input spatial extent.
    ///
    /// The valid extent is
    /// `floor((input + 2*padding - effective_kernel) / stride) + 1` when the
    /// padded input can contain the effective kernel, and zero otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] for shape arithmetic overflow.
    pub fn output_extent(&self, axis: usize, input_extent: usize) -> Result<usize> {
        let padding = *self.padding.get(axis).ok_or_else(|| {
            LetoError::InvalidInput(format!("window axis {axis} exceeds rank {D}"))
        })?;
        let stride = *self.stride.get(axis).ok_or_else(|| {
            LetoError::InvalidInput(format!("window axis {axis} exceeds rank {D}"))
        })?;
        let padded_input = padding
            .checked_mul(2)
            .and_then(|value| input_extent.checked_add(value))
            .ok_or(LetoError::Overflow {
                reason: "window padded input extent",
            })?;
        let effective_kernel = self.effective_kernel_extent(axis)?;
        padded_input
            .checked_sub(effective_kernel)
            .map_or(Ok(0), |extent| {
                extent
                    .checked_div(stride)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(LetoError::Overflow {
                        reason: "window output extent",
                    })
            })
    }

    /// Derive all output spatial extents from input spatial extents.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] for shape arithmetic overflow.
    pub fn output_shape(&self, input_spatial: [usize; D]) -> Result<[usize; D]> {
        let mut output = [0; D];
        for (axis, &extent) in input_spatial.iter().enumerate() {
            output[axis] = self.output_extent(axis, extent)?;
        }
        Ok(output)
    }

    /// Return the number of points in the window.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::Overflow`] when the product exceeds `usize`.
    pub fn kernel_volume(&self) -> Result<usize> {
        self.kernel.iter().try_fold(1_usize, |volume, &extent| {
            volume.checked_mul(extent).ok_or(LetoError::Overflow {
                reason: "window kernel volume",
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::WindowParameters;
    use crate::LetoError;

    #[test]
    fn rejects_zero_geometry() {
        assert_eq!(
            WindowParameters::new([0], [1], [0], [1]),
            Err(LetoError::InvalidInput(
                "window kernel extents must be nonzero".to_string()
            ))
        );
        assert_eq!(
            WindowParameters::new([1], [0], [0], [1]),
            Err(LetoError::InvalidInput(
                "window stride extents must be nonzero".to_string()
            ))
        );
    }

    #[test]
    fn derives_anisotropic_output_shape() {
        let parameters = WindowParameters::new([2, 3], [2, 1], [1, 0], [1, 2])
            .expect("valid spatial window parameters");
        assert_eq!(parameters.effective_kernel_extent(0), Ok(2));
        assert_eq!(parameters.effective_kernel_extent(1), Ok(5));
        assert_eq!(parameters.output_shape([5, 8]), Ok([3, 4]));
        assert_eq!(parameters.kernel_volume(), Ok(6));
    }
}
