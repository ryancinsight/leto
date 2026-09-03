use crate::domain::error::Result;
use crate::domain::layout::kernels;

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Broadcast the current layout to a target shape of length `M` where `M >= N`.
    pub fn broadcast<const M: usize>(&self, target_shape: [usize; M]) -> Result<Layout<M>> {
        let mut new_strides = [0isize; M];
        kernels::broadcast_strides(&self.shape, &self.strides, &target_shape, &mut new_strides)?;
        Layout::try_new(target_shape, new_strides, self.offset)
    }
}
