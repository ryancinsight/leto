use super::sources::{zip_offset, ZipSources};
use super::traversal::for_each_row_major_indexed;
use crate::application::index::{validate_mutable_output, RowMajorTraversal};
use leto::{ArrayViewMut, LetoError, Result};

mod sealed {
    pub trait Sealed<S, F, const N: usize> {}
}

#[inline]
fn validate_output<T, const N: usize>(output: &ArrayViewMut<'_, T, N>) -> Result<()> {
    validate_mutable_output(output, "zip mutable")
}

fn zip_one<T, S, F, const N: usize>(
    output: &mut ArrayViewMut<'_, T, N>,
    sources: S,
    mut f: F,
) -> Result<()>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    validate_output(output)?;
    let shape = output.shape();
    sources.validate(shape)?;
    if let (Some(output_slice), Some(source_slices)) = (output.as_mut_slice(), sources.contiguous())
    {
        for (index, value) in output_slice.iter_mut().enumerate() {
            f(value, S::contiguous_values(source_slices, index));
        }
        return Ok(());
    }
    let size = output.layout().checked_size()?;
    let output_layout = output.layout();
    let output_data = output.data_mut();
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let output_step = traversal.last_axis_stride(output_layout);
    let source_steps = sources.steps();

    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut output_offset = zip_offset(output_layout, base)?;
        let mut source_offsets = sources.offsets_at(base)?;
        for _ in 0..traversal.inner() {
            let output_index = usize::try_from(output_offset)
                .expect("invariant: validated zip output offset is non-negative");
            f(
                &mut output_data[output_index],
                sources.values(source_offsets),
            );
            output_offset += output_step;
            sources.advance(&mut source_offsets, source_steps);
        }
    }
    Ok(())
}

fn indexed_zip_one<T, S, F, const N: usize>(
    output: &mut ArrayViewMut<'_, T, N>,
    sources: S,
    mut f: F,
) -> Result<()>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    validate_output(output)?;
    let shape = output.shape();
    sources.validate(shape)?;
    let size = output.layout().checked_size()?;
    let output_layout = output.layout();
    let output_data = output.data_mut();
    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let output_step = traversal.last_axis_stride(output_layout);
    let source_steps = sources.steps();

    for_each_row_major_indexed(
        traversal,
        |index| {
            Ok((
                zip_offset(output_layout, index)?,
                sources.offsets_at(index)?,
            ))
        },
        |index, offsets| {
            let output_index = usize::try_from(offsets.0)
                .expect("invariant: validated zip output offset is non-negative");
            f(
                index,
                &mut output_data[output_index],
                sources.values(offsets.1),
            );
            offsets.0 += output_step;
            sources.advance(&mut offsets.1, source_steps);
        },
    )?;
    Ok(())
}

/// A statically typed mutable-output zip operation.
///
/// Implementations cover one, two, and three mutable views. The operation
/// trait carries the source and closure types so the closure receives concrete
/// mutable references at each callsite without a higher-ranked trait bound over
/// a generic associated lifetime.
pub trait ZipMutOutputs<S, F, const N: usize>: sealed::Sealed<S, F, N> {
    /// Mutate the output views with the source values.
    fn zip(outputs: Self, sources: S, f: F) -> Result<()>;
}

/// Indexed counterpart to [`ZipMutOutputs`].
pub trait IndexedZipMutOutputs<S, F, const N: usize>: sealed::Sealed<S, F, N> {
    /// Mutate the output views with source values and logical indices.
    fn indexed_zip(outputs: Self, sources: S, f: F) -> Result<()>;
}

impl<'data, T, S, F, const N: usize> sealed::Sealed<S, F, N> for ArrayViewMut<'data, T, N> {}

impl<'data, T, S, F, const N: usize> ZipMutOutputs<S, F, N> for ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    fn zip(mut output: Self, sources: S, f: F) -> Result<()> {
        zip_one(&mut output, sources, f)
    }
}

macro_rules! impl_zip_mut_outputs_for_tuple {
    ($($type:ident => $data:ident : $index:tt),+ $(,)?) => {
        impl<'data, $($type,)+ S, F, const N: usize>
            sealed::Sealed<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        {
        }

        impl<'data, $($type,)+ S, F, const N: usize>
            ZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        where
            S: ZipSources<N>,
            F: FnMut(($(&mut $type,)+), S::Values),
        {
            fn zip(mut outputs: Self, sources: S, mut f: F) -> Result<()> {
                let shape = outputs.0.shape();
                $(
                    if outputs.$index.shape() != shape {
                        return Err(LetoError::ShapeMismatch {
                            lhs: shape.to_vec(),
                            rhs: outputs.$index.shape().to_vec(),
                        });
                    }
                    validate_output(&outputs.$index)?;
                )+
                let size = outputs.0.layout().checked_size()?;
                if let ($(Some($data),)+) = ($(outputs.$index.as_mut_slice(),)+) {
                    if let Some(source_slices) = sources.contiguous() {
                        for index in 0..size {
                            f(
                                ($(&mut $data[index],)+),
                                S::contiguous_values(source_slices, index),
                            );
                        }
                        return Ok(());
                    }
                }
                let output_layouts = ($(outputs.$index.layout(),)+);
                let ($($data,)+) = ($(outputs.$index.data_mut(),)+);
                let Some(traversal) = RowMajorTraversal::new(size, shape) else {
                    return Ok(());
                };
                let output_steps = ($(traversal.last_axis_stride(output_layouts.$index),)+);
                let source_steps = sources.steps();

                for row in 0..traversal.rows() {
                    let base = traversal.base_index(row);
                    let mut output_offsets =
                        ($(zip_offset(output_layouts.$index, base)?,)+);
                    let mut source_offsets = sources.offsets_at(base)?;
                    for _ in 0..traversal.inner() {
                        f(
                            ($(&mut $data[usize::try_from(output_offsets.$index)
                                .expect("invariant: validated zip output offset is non-negative")],)+),
                            sources.values(source_offsets),
                        );
                        $(output_offsets.$index += output_steps.$index;)+
                        sources.advance(&mut source_offsets, source_steps);
                    }
                }
                Ok(())
            }

        }
    };
}

impl_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1);
impl_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1, C => c: 2);

impl<'a, 'data, T, S, F, const N: usize> sealed::Sealed<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
{
}

impl<'a, 'data, T, S, F, const N: usize> ZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut(&mut T, S::Values),
{
    fn zip(outputs: Self, sources: S, f: F) -> Result<()> {
        zip_one(outputs, sources, f)
    }
}

impl<'data, T, S, F, const N: usize> IndexedZipMutOutputs<S, F, N> for ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    fn indexed_zip(mut output: Self, sources: S, f: F) -> Result<()> {
        indexed_zip_one(&mut output, sources, f)
    }
}

macro_rules! impl_indexed_zip_mut_outputs_for_tuple {
    ($($type:ident => $data:ident : $index:tt),+ $(,)?) => {
        impl<'data, $($type,)+ S, F, const N: usize>
            IndexedZipMutOutputs<S, F, N>
            for ($(ArrayViewMut<'data, $type, N>,)+)
        where
            S: ZipSources<N>,
            F: FnMut([usize; N], ($(&mut $type,)+), S::Values),
        {
            fn indexed_zip(mut outputs: Self, sources: S, mut f: F) -> Result<()> {
                let shape = outputs.0.shape();
                $(
                    if outputs.$index.shape() != shape {
                        return Err(LetoError::ShapeMismatch {
                            lhs: shape.to_vec(),
                            rhs: outputs.$index.shape().to_vec(),
                        });
                    }
                    validate_output(&outputs.$index)?;
                )+
                let size = outputs.0.layout().checked_size()?;
                let output_layouts = ($(outputs.$index.layout(),)+);
                let ($($data,)+) = ($(outputs.$index.data_mut(),)+);
                let Some(traversal) = RowMajorTraversal::new(size, shape) else {
                    return Ok(());
                };
                let output_steps = ($(traversal.last_axis_stride(output_layouts.$index),)+);
                let source_steps = sources.steps();

                for_each_row_major_indexed(
                    traversal,
                    |index| {
                        Ok((
                            ($(zip_offset(output_layouts.$index, index)?,)+),
                            sources.offsets_at(index)?,
                        ))
                    },
                    |index, state| {
                        let (output_offsets, source_offsets) = state;
                        f(
                            index,
                            ($(&mut $data[usize::try_from(output_offsets.$index)
                                .expect("invariant: validated zip output offset is non-negative")],)+),
                            sources.values(*source_offsets),
                        );
                        $(output_offsets.$index += output_steps.$index;)+
                        sources.advance(source_offsets, source_steps);
                    },
                )?;
                Ok(())
            }
        }
    };
}

impl_indexed_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1);
impl_indexed_zip_mut_outputs_for_tuple!(A => a: 0, B => b: 1, C => c: 2);

impl<'a, 'data, T, S, F, const N: usize> IndexedZipMutOutputs<S, F, N>
    for &'a mut ArrayViewMut<'data, T, N>
where
    S: ZipSources<N>,
    F: FnMut([usize; N], &mut T, S::Values),
{
    fn indexed_zip(outputs: Self, sources: S, f: F) -> Result<()> {
        indexed_zip_one(outputs, sources, f)
    }
}

/// Mutably zip-map one or more output views with zero or more read-only views.
///
/// A single output is passed as `&mut T`; multiple outputs are passed as a
/// tuple. Read-only sources follow the same rule: one source is passed directly
/// and multiple sources are passed as a tuple. All arities are statically
/// dispatched, so the output/source family and closure monomorphize together.
pub fn zip_mut_with<O, S, F, const N: usize>(outputs: O, sources: S, f: F) -> Result<()>
where
    O: ZipMutOutputs<S, F, N>,
{
    O::zip(outputs, sources, f)
}

/// Mutably zip-map output views with zero or more read-only views and logical
/// row-major indices.
pub fn indexed_zip_mut_with<O, S, F, const N: usize>(outputs: O, sources: S, f: F) -> Result<()>
where
    O: IndexedZipMutOutputs<S, F, N>,
{
    O::indexed_zip(outputs, sources, f)
}
