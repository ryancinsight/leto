use crate::domain::error::Result;

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Broadcast the current layout to a target shape of length `M` where `M >= N`.
    pub fn broadcast<const M: usize>(&self, target_shape: [usize; M]) -> Result<Layout<M>> {
        let mut new_shape = [0usize; M];
        let mut new_strides = [0isize; M];
        crate::domain::layout::kernels::broadcast_layout(
            &self.shape,
            &self.strides,
            &target_shape,
            &mut new_shape,
            &mut new_strides,
        )?;
        Layout::try_new(new_shape, new_strides, self.offset)
    }
}
