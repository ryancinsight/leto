use crate::domain::error::{LetoError, Result};

use super::Layout;

impl<const N: usize> Layout<N> {
    /// Broadcast the current layout to a target shape of length `M` where `M >= N`.
    pub fn broadcast<const M: usize>(&self, target_shape: [usize; M]) -> Result<Layout<M>> {
        if M < N {
            return Err(LetoError::IncompatibleBroadcast {
                from: self.shape.to_vec(),
                to: target_shape.to_vec(),
            });
        }

        let mut new_shape = [0usize; M];
        let mut new_strides = [0isize; M];
        let shift = M - N;

        // Populate prepended dimensions
        for i in 0..shift {
            new_shape[i] = target_shape[i];
            new_strides[i] = 0; // Stride is 0 for broadcasted dimensions
        }

        // Populate matching dimensions
        for i in 0..N {
            let target_idx = i + shift;
            let target_dim = target_shape[target_idx];
            let source_dim = self.shape[i];

            if source_dim == target_dim {
                new_shape[target_idx] = target_dim;
                new_strides[target_idx] = self.strides[i];
            } else if source_dim == 1 {
                new_shape[target_idx] = target_dim;
                new_strides[target_idx] = 0; // Stride is 0 when broadcasting a 1-sized dim
            } else {
                return Err(LetoError::IncompatibleBroadcast {
                    from: self.shape.to_vec(),
                    to: target_shape.to_vec(),
                });
            }
        }

        Layout::try_new(new_shape, new_strides, self.offset)
    }
}
