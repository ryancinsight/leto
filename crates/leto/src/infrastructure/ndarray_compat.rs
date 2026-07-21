use crate::application::array::Array;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::VecStorage;
use ndarray::{Dimension, IntoDimension, ShapeBuilder};

/// Re-export of `ndarray` for consumer crates that need boundary I/O interop
/// without declaring a direct `ndarray` dependency.
pub use ndarray;

fn ndarray_relative_span<const N: usize>(
    shape: &[usize],
    strides: &[isize],
) -> Result<(isize, isize)> {
    let mut min_offset = 0isize;
    let mut max_offset = 0isize;

    for axis in 0..N {
        if shape[axis] == 0 {
            return Ok((0, 0));
        }

        let extent = isize::try_from(shape[axis] - 1).map_err(|_| LetoError::Overflow {
            reason: "ndarray dimension extent conversion",
        })?;
        let delta = extent
            .checked_mul(strides[axis])
            .ok_or(LetoError::Overflow {
                reason: "ndarray view span multiplication",
            })?;
        min_offset = min_offset
            .checked_add(delta.min(0))
            .ok_or(LetoError::Overflow {
                reason: "ndarray view minimum span accumulation",
            })?;
        max_offset = max_offset
            .checked_add(delta.max(0))
            .ok_or(LetoError::Overflow {
                reason: "ndarray view maximum span accumulation",
            })?;
    }

    Ok((min_offset, max_offset))
}

fn leto_layout_from_ndarray<const N: usize>(
    shape: &[usize],
    strides: &[isize],
) -> Result<(Layout<N>, usize)> {
    let (min_offset, max_offset) = ndarray_relative_span::<N>(shape, strides)?;
    let span_len = max_offset
        .checked_sub(min_offset)
        .and_then(|span| span.checked_add(1))
        .ok_or(LetoError::Overflow {
            reason: "ndarray view span length",
        })?;
    let span_len = usize::try_from(span_len).map_err(|_| LetoError::Overflow {
        reason: "ndarray view span length conversion",
    })?;
    let base_offset = usize::try_from(-min_offset).map_err(|_| LetoError::Overflow {
        reason: "ndarray view base offset conversion",
    })?;

    let mut leto_shape = [0usize; N];
    let mut leto_strides = [0isize; N];
    leto_shape[..N].copy_from_slice(&shape[..N]);
    leto_strides[..N].copy_from_slice(&strides[..N]);

    Ok((Layout::new(leto_shape, leto_strides, base_offset), span_len))
}

impl<'a, T, const N: usize> TryFrom<ArrayView<'a, T, N>>
    for ndarray::ArrayView<'a, T, ndarray::Dim<[usize; N]>>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    type Error = LetoError;

    #[inline]
    fn try_from(view: ArrayView<'a, T, N>) -> Result<Self> {
        let mut strides_usize = [0usize; N];
        let strides = view.strides();
        for (i, &stride) in strides.iter().enumerate() {
            if stride < 0 {
                return Err(LetoError::StorageError {
                    reason: format!(
                        "Cannot convert Leto ArrayView with negative stride {} to ndarray",
                        stride
                    ),
                });
            }
            strides_usize[i] = stride as usize;
        }

        let nd_shape = view
            .shape()
            .into_dimension()
            .strides(strides_usize.into_dimension());
        let offset = view.offset();
        let slice = &view.data[offset..];
        let nd_view = ndarray::ArrayView::from_shape(nd_shape, slice).map_err(|e| {
            LetoError::StorageError {
                reason: e.to_string(),
            }
        })?;
        Ok(nd_view)
    }
}

impl<'a, T, const N: usize> TryFrom<ArrayViewMut<'a, T, N>>
    for ndarray::ArrayViewMut<'a, T, ndarray::Dim<[usize; N]>>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    type Error = LetoError;

    #[inline]
    fn try_from(view: ArrayViewMut<'a, T, N>) -> Result<Self> {
        let mut strides_usize = [0usize; N];
        let strides = view.strides();
        for (i, &stride) in strides.iter().enumerate() {
            if stride < 0 {
                return Err(LetoError::StorageError {
                    reason: format!(
                        "Cannot convert Leto ArrayViewMut with negative stride {} to ndarray",
                        stride
                    ),
                });
            }
            strides_usize[i] = stride as usize;
        }

        let nd_shape = view
            .shape()
            .into_dimension()
            .strides(strides_usize.into_dimension());
        let offset = view.offset();
        let slice = &mut view.into_slice()[offset..];
        let nd_view = ndarray::ArrayViewMut::from_shape(nd_shape, slice).map_err(|e| {
            LetoError::StorageError {
                reason: e.to_string(),
            }
        })?;
        Ok(nd_view)
    }
}

impl<'a, T, const N: usize> From<ndarray::ArrayView<'a, T, ndarray::Dim<[usize; N]>>>
    for ArrayView<'a, T, N>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    #[inline]
    fn from(nd_view: ndarray::ArrayView<'a, T, ndarray::Dim<[usize; N]>>) -> Self {
        let nd_shape = nd_view.shape();
        let nd_strides = nd_view.strides();
        let (layout, span_len) = leto_layout_from_ndarray::<N>(nd_shape, nd_strides)
            .expect("ndarray view span must fit in isize and usize");
        let (min_offset, _) = ndarray_relative_span::<N>(nd_shape, nd_strides)
            .expect("ndarray view span must fit in isize");

        let data = if nd_shape.contains(&0) {
            &[]
        } else {
            // SAFETY: `min_offset..=max_offset` is the physical span covered by this
            // ndarray view. The resulting slice starts at the minimum address and
            // covers every element reachable by the Leto layout.
            unsafe { std::slice::from_raw_parts(nd_view.as_ptr().offset(min_offset), span_len) }
        };

        Self::new(layout, data)
    }
}

impl<'a, T, const N: usize> From<ndarray::ArrayViewMut<'a, T, ndarray::Dim<[usize; N]>>>
    for ArrayViewMut<'a, T, N>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    #[inline]
    fn from(mut nd_view: ndarray::ArrayViewMut<'a, T, ndarray::Dim<[usize; N]>>) -> Self {
        let nd_shape = nd_view.shape();
        let nd_strides = nd_view.strides();
        let (layout, span_len) = leto_layout_from_ndarray::<N>(nd_shape, nd_strides)
            .expect("ndarray mutable view span must fit in isize and usize");
        let (min_offset, _) = ndarray_relative_span::<N>(nd_shape, nd_strides)
            .expect("ndarray mutable view span must fit in isize");

        let data = if nd_shape.contains(&0) {
            &mut []
        } else {
            // SAFETY: `min_offset..=max_offset` is the physical span covered by this
            // ndarray view. The resulting slice starts at the minimum address and
            // covers every element reachable by the Leto layout.
            unsafe {
                std::slice::from_raw_parts_mut(nd_view.as_mut_ptr().offset(min_offset), span_len)
            }
        };

        Self::new(layout, data)
    }
}

impl<T, const N: usize> TryFrom<Array<T, VecStorage<T>, N>>
    for ndarray::Array<T, ndarray::Dim<[usize; N]>>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    type Error = LetoError;

    #[inline]
    fn try_from(array: Array<T, VecStorage<T>, N>) -> Result<Self> {
        if !array.layout().is_c_contiguous() {
            return Err(LetoError::StorageError {
                reason: "Cannot convert non-C-contiguous Leto Array to ndarray::Array without copy"
                    .to_string(),
            });
        }
        let shape = array.shape();
        let vec = array.storage.into_inner();
        let nd_array =
            ndarray::Array::from_shape_vec(shape.into_dimension(), vec).map_err(|e| {
                LetoError::StorageError {
                    reason: e.to_string(),
                }
            })?;
        Ok(nd_array)
    }
}

impl<T: Clone, const N: usize> From<ndarray::Array<T, ndarray::Dim<[usize; N]>>>
    for Array<T, VecStorage<T>, N>
where
    [usize; N]: IntoDimension<Dim = ndarray::Dim<[usize; N]>>,
    ndarray::Dim<[usize; N]>: Dimension,
{
    #[inline]
    fn from(nd_array: ndarray::Array<T, ndarray::Dim<[usize; N]>>) -> Self {
        let is_contiguous = nd_array.is_standard_layout();
        let shape = nd_array.shape();
        let strides = nd_array.strides();

        let mut l_shape = [0usize; N];
        let mut l_strides = [0isize; N];
        l_shape[..N].copy_from_slice(&shape[..N]);
        l_strides[..N].copy_from_slice(&strides[..N]);

        if is_contiguous {
            let (vec, _) = nd_array.into_raw_vec_and_offset();
            let layout = Layout::new(l_shape, l_strides, 0);
            Self::new(layout, VecStorage::new(vec)).expect("ndarray layout is valid")
        } else {
            let contiguous = nd_array.to_owned();
            let (vec, _) = contiguous.into_raw_vec_and_offset();
            let layout = Layout::c_contiguous(l_shape).expect("ndarray shape produces valid c-contiguous layout");
            Self::new(layout, VecStorage::new(vec)).expect("ndarray layout is valid")
        }
    }
}
