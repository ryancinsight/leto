use crate::domain::error::{LetoError, Result};

/// A trait mapping dimension rank `N` to `N - 1` at compile-time on stable Rust.
pub trait RemoveAxis<const N: usize> {
    /// The resulting rank after removing one axis (always `N - 1`).
    const SMALLER_RANK: usize;

    /// The shape type of the reduced layout (always `[usize; N - 1]`).
    type SmallerShape: Copy + Send + Sync + 'static + AsRef<[usize]> + AsMut<[usize]>;

    /// The strides type of the reduced layout (always `[isize; N - 1]`).
    type SmallerStrides: Copy + Send + Sync + 'static + AsRef<[isize]> + AsMut<[isize]>;

    /// Remove an axis from a shape array.
    fn remove_shape(&self, shape: [usize; N], axis: usize) -> Result<Self::SmallerShape>;

    /// Remove an axis from a strides array.
    fn remove_strides(&self, strides: [isize; N], axis: usize) -> Result<Self::SmallerStrides>;
}

/// A zero-sized type marker representing a compile-time rank.
pub struct RankMarker<const N: usize>;

impl RemoveAxis<1> for RankMarker<1> {
    const SMALLER_RANK: usize = 0;
    type SmallerShape = [usize; 0];
    type SmallerStrides = [isize; 0];

    #[inline]
    fn remove_shape(&self, _shape: [usize; 1], axis: usize) -> Result<Self::SmallerShape> {
        if axis != 0 {
            return Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 1"),
            });
        }
        Ok([])
    }

    #[inline]
    fn remove_strides(&self, _strides: [isize; 1], axis: usize) -> Result<Self::SmallerStrides> {
        if axis != 0 {
            return Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 1"),
            });
        }
        Ok([])
    }
}

impl RemoveAxis<2> for RankMarker<2> {
    const SMALLER_RANK: usize = 1;
    type SmallerShape = [usize; 1];
    type SmallerStrides = [isize; 1];

    #[inline]
    fn remove_shape(&self, shape: [usize; 2], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1]]),
            1 => Ok([shape[0]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 2"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 2], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([strides[1]]),
            1 => Ok([strides[0]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 2"),
            }),
        }
    }
}

impl RemoveAxis<3> for RankMarker<3> {
    const SMALLER_RANK: usize = 2;
    type SmallerShape = [usize; 2];
    type SmallerStrides = [isize; 2];

    #[inline]
    fn remove_shape(&self, shape: [usize; 3], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1], shape[2]]),
            1 => Ok([shape[0], shape[2]]),
            2 => Ok([shape[0], shape[1]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 3"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 3], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([strides[1], strides[2]]),
            1 => Ok([strides[0], strides[2]]),
            2 => Ok([strides[0], strides[1]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 3"),
            }),
        }
    }
}

impl RemoveAxis<4> for RankMarker<4> {
    const SMALLER_RANK: usize = 3;
    type SmallerShape = [usize; 3];
    type SmallerStrides = [isize; 3];

    #[inline]
    fn remove_shape(&self, shape: [usize; 4], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1], shape[2], shape[3]]),
            1 => Ok([shape[0], shape[2], shape[3]]),
            2 => Ok([shape[0], shape[1], shape[3]]),
            3 => Ok([shape[0], shape[1], shape[2]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 4"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 4], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([strides[1], strides[2], strides[3]]),
            1 => Ok([strides[0], strides[2], strides[3]]),
            2 => Ok([strides[0], strides[1], strides[3]]),
            3 => Ok([strides[0], strides[1], strides[2]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 4"),
            }),
        }
    }
}

impl RemoveAxis<5> for RankMarker<5> {
    const SMALLER_RANK: usize = 4;
    type SmallerShape = [usize; 4];
    type SmallerStrides = [isize; 4];

    #[inline]
    fn remove_shape(&self, shape: [usize; 5], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1], shape[2], shape[3], shape[4]]),
            1 => Ok([shape[0], shape[2], shape[3], shape[4]]),
            2 => Ok([shape[0], shape[1], shape[3], shape[4]]),
            3 => Ok([shape[0], shape[1], shape[2], shape[4]]),
            4 => Ok([shape[0], shape[1], shape[2], shape[3]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 5"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 5], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([strides[1], strides[2], strides[3], strides[4]]),
            1 => Ok([strides[0], strides[2], strides[3], strides[4]]),
            2 => Ok([strides[0], strides[1], strides[3], strides[4]]),
            3 => Ok([strides[0], strides[1], strides[2], strides[4]]),
            4 => Ok([strides[0], strides[1], strides[2], strides[3]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 5"),
            }),
        }
    }
}

impl RemoveAxis<6> for RankMarker<6> {
    const SMALLER_RANK: usize = 5;
    type SmallerShape = [usize; 5];
    type SmallerStrides = [isize; 5];

    #[inline]
    fn remove_shape(&self, shape: [usize; 6], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1], shape[2], shape[3], shape[4], shape[5]]),
            1 => Ok([shape[0], shape[2], shape[3], shape[4], shape[5]]),
            2 => Ok([shape[0], shape[1], shape[3], shape[4], shape[5]]),
            3 => Ok([shape[0], shape[1], shape[2], shape[4], shape[5]]),
            4 => Ok([shape[0], shape[1], shape[2], shape[3], shape[5]]),
            5 => Ok([shape[0], shape[1], shape[2], shape[3], shape[4]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 6"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 6], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([strides[1], strides[2], strides[3], strides[4], strides[5]]),
            1 => Ok([strides[0], strides[2], strides[3], strides[4], strides[5]]),
            2 => Ok([strides[0], strides[1], strides[3], strides[4], strides[5]]),
            3 => Ok([strides[0], strides[1], strides[2], strides[4], strides[5]]),
            4 => Ok([strides[0], strides[1], strides[2], strides[3], strides[5]]),
            5 => Ok([strides[0], strides[1], strides[2], strides[3], strides[4]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 6"),
            }),
        }
    }
}

impl RemoveAxis<7> for RankMarker<7> {
    const SMALLER_RANK: usize = 6;
    type SmallerShape = [usize; 6];
    type SmallerStrides = [isize; 6];

    #[inline]
    fn remove_shape(&self, shape: [usize; 7], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]]),
            1 => Ok([shape[0], shape[2], shape[3], shape[4], shape[5], shape[6]]),
            2 => Ok([shape[0], shape[1], shape[3], shape[4], shape[5], shape[6]]),
            3 => Ok([shape[0], shape[1], shape[2], shape[4], shape[5], shape[6]]),
            4 => Ok([shape[0], shape[1], shape[2], shape[3], shape[5], shape[6]]),
            5 => Ok([shape[0], shape[1], shape[2], shape[3], shape[4], shape[6]]),
            6 => Ok([shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 7"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 7], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([
                strides[1], strides[2], strides[3], strides[4], strides[5], strides[6],
            ]),
            1 => Ok([
                strides[0], strides[2], strides[3], strides[4], strides[5], strides[6],
            ]),
            2 => Ok([
                strides[0], strides[1], strides[3], strides[4], strides[5], strides[6],
            ]),
            3 => Ok([
                strides[0], strides[1], strides[2], strides[4], strides[5], strides[6],
            ]),
            4 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[5], strides[6],
            ]),
            5 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[4], strides[6],
            ]),
            6 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[4], strides[5],
            ]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 7"),
            }),
        }
    }
}

impl RemoveAxis<8> for RankMarker<8> {
    const SMALLER_RANK: usize = 7;
    type SmallerShape = [usize; 7];
    type SmallerStrides = [isize; 7];

    #[inline]
    fn remove_shape(&self, shape: [usize; 8], axis: usize) -> Result<Self::SmallerShape> {
        match axis {
            0 => Ok([
                shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
            ]),
            1 => Ok([
                shape[0], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
            ]),
            2 => Ok([
                shape[0], shape[1], shape[3], shape[4], shape[5], shape[6], shape[7],
            ]),
            3 => Ok([
                shape[0], shape[1], shape[2], shape[4], shape[5], shape[6], shape[7],
            ]),
            4 => Ok([
                shape[0], shape[1], shape[2], shape[3], shape[5], shape[6], shape[7],
            ]),
            5 => Ok([
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[6], shape[7],
            ]),
            6 => Ok([
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[7],
            ]),
            7 => Ok([
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
            ]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 8"),
            }),
        }
    }

    #[inline]
    fn remove_strides(&self, strides: [isize; 8], axis: usize) -> Result<Self::SmallerStrides> {
        match axis {
            0 => Ok([
                strides[1], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
            ]),
            1 => Ok([
                strides[0], strides[2], strides[3], strides[4], strides[5], strides[6], strides[7],
            ]),
            2 => Ok([
                strides[0], strides[1], strides[3], strides[4], strides[5], strides[6], strides[7],
            ]),
            3 => Ok([
                strides[0], strides[1], strides[2], strides[4], strides[5], strides[6], strides[7],
            ]),
            4 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[5], strides[6], strides[7],
            ]),
            5 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[4], strides[6], strides[7],
            ]),
            6 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[7],
            ]),
            7 => Ok([
                strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6],
            ]),
            _ => Err(LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank 8"),
            }),
        }
    }
}
