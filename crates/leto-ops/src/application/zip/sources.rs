use leto::{ArrayView, Layout, LetoError, Result};

mod sealed {
    pub trait Sealed<const N: usize> {}
}

/// A statically typed set of read-only views for a multi-input zip.
///
/// Implementations are provided for read-only ArrayView references and
/// tuples containing them. Tuple values are passed to the closure as one
/// nested value, so heterogeneous element types remain statically dispatched
/// without allocating a type-erased source list.
pub trait ZipSources<const N: usize>: sealed::Sealed<N> {
    /// Values borrowed from the source set at one logical element.
    type Values;

    /// Physical offsets for the source set at one logical element.
    type Offsets: Copy;

    /// Contiguous source slices used by the dense traversal fast path.
    type Contiguous: Copy;

    /// Validate source storage and require the expected logical shape.
    fn validate(&self, expected_shape: [usize; N]) -> Result<()>;

    /// Compute physical source offsets for one logical index.
    fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets>;

    /// Return the last-axis stride for each source.
    fn steps(&self) -> Self::Offsets;

    /// Advance all source offsets by one logical element.
    fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets);

    /// Return dense row-major slices when every source is contiguous.
    fn contiguous(&self) -> Option<Self::Contiguous>;

    /// Borrow source values from dense slices at one logical position.
    fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values;

    /// Borrow source values from strided storage at the given offsets.
    fn values(&self, offsets: Self::Offsets) -> Self::Values;
}

#[inline]
pub(super) fn zip_offset<const N: usize>(layout: Layout<N>, index: [usize; N]) -> Result<isize> {
    isize::try_from(layout.offset_of(index)?).map_err(|_| LetoError::StorageError {
        reason: "zip layout offset exceeds isize range".to_string(),
    })
}

impl<'data, T, const N: usize> sealed::Sealed<N> for &ArrayView<'data, T, N> {}

impl<'data, T, const N: usize> ZipSources<N> for &ArrayView<'data, T, N> {
    type Values = &'data T;
    type Offsets = isize;
    type Contiguous = &'data [T];

    #[inline]
    fn validate(&self, expected_shape: [usize; N]) -> Result<()> {
        let view = *self;
        if view.shape() != expected_shape {
            return Err(LetoError::ShapeMismatch {
                lhs: expected_shape.to_vec(),
                rhs: view.shape().to_vec(),
            });
        }
        view.layout().validate_storage_len(view.data().len())
    }

    #[inline]
    fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets> {
        zip_offset((*self).layout(), index)
    }

    #[inline]
    fn steps(&self) -> Self::Offsets {
        if N == 0 {
            0
        } else {
            (*self).layout().strides()[N - 1]
        }
    }

    #[inline]
    fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets) {
        *offsets += steps;
    }

    #[inline]
    fn contiguous(&self) -> Option<Self::Contiguous> {
        (*self).as_slice()
    }

    #[inline]
    fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values {
        &sources[index]
    }

    #[inline]
    fn values(&self, offsets: Self::Offsets) -> Self::Values {
        let view = *self;
        let offset =
            usize::try_from(offsets).expect("invariant: validated zip offset is non-negative");
        &view.data()[offset]
    }
}

impl<const N: usize> sealed::Sealed<N> for () {}

impl<const N: usize> ZipSources<N> for () {
    type Values = ();
    type Offsets = ();
    type Contiguous = ();

    #[inline]
    fn validate(&self, _expected_shape: [usize; N]) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn offsets_at(&self, _index: [usize; N]) -> Result<Self::Offsets> {
        Ok(())
    }

    #[inline]
    fn steps(&self) -> Self::Offsets {}

    #[inline]
    fn advance(&self, _offsets: &mut Self::Offsets, _steps: Self::Offsets) {}

    #[inline]
    fn contiguous(&self) -> Option<Self::Contiguous> {
        Some(())
    }

    #[inline]
    fn contiguous_values(_sources: Self::Contiguous, _index: usize) -> Self::Values {}

    #[inline]
    fn values(&self, _offsets: Self::Offsets) -> Self::Values {}
}

macro_rules! impl_zip_sources_for_tuple {
    ($($source:ident : $index:tt),+ $(,)?) => {
        impl<$($source,)+ const N: usize> sealed::Sealed<N> for ($($source,)+)
        where
            $($source: ZipSources<N>,)+
        {
        }

        impl<$($source,)+ const N: usize> ZipSources<N> for ($($source,)+)
        where
            $($source: ZipSources<N>,)+
        {
            type Values = ($($source::Values,)+);
            type Offsets = ($($source::Offsets,)+);
            type Contiguous = ($($source::Contiguous,)+);

            #[inline]
            fn validate(&self, expected_shape: [usize; N]) -> Result<()> {
                $(self.$index.validate(expected_shape)?;)+
                Ok(())
            }

            #[inline]
            fn offsets_at(&self, index: [usize; N]) -> Result<Self::Offsets> {
                Ok(($(self.$index.offsets_at(index)?,)+))
            }

            #[inline]
            fn steps(&self) -> Self::Offsets {
                ($(self.$index.steps(),)+)
            }

            #[inline]
            fn advance(&self, offsets: &mut Self::Offsets, steps: Self::Offsets) {
                $(self.$index.advance(&mut offsets.$index, steps.$index);)+
            }

            #[inline]
            fn contiguous(&self) -> Option<Self::Contiguous> {
                Some(($(self.$index.contiguous()?,)+))
            }

            #[inline]
            fn contiguous_values(sources: Self::Contiguous, index: usize) -> Self::Values {
                ($(<$source as ZipSources<N>>::contiguous_values(
                    sources.$index,
                    index,
                ),)+)
            }

            #[inline]
            fn values(&self, offsets: Self::Offsets) -> Self::Values {
                ($(self.$index.values(offsets.$index),)+)
            }
        }
    };
}

impl_zip_sources_for_tuple!(A: 0, B: 1);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
impl_zip_sources_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);
