use super::traversal::for_each_row_major_indexed;
use crate::application::index::{validate_mutable_output, RowMajorTraversal};
use leto::{ArrayViewMut, LetoError, Result};

/// Mutably map elements in place with the logical row-major index.
///
/// This is the one-view indexed analogue of
/// [`crate::application::zip::zip_mut_with`]. Every logical element is visited
/// exactly once, and the closure receives the logical index before the
/// mutable element.
pub fn indexed_map_inplace<T, F, const N: usize>(
    view: &mut ArrayViewMut<'_, T, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut T),
{
    validate_mutable_output(view, "indexed mutable map")?;

    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let step = traversal.last_axis_stride(layout);
    for_each_row_major_indexed(
        traversal,
        |index| Ok(layout.offset_of(index)? as isize),
        |index, offset| {
            f(index, &mut data[*offset as usize]);
            *offset += step;
        },
    )?;

    Ok(())
}

/// Mutably map four views in place with the logical row-major index.
///
/// This is the multi-output analogue of [`indexed_map_inplace`]. All four views
/// must share the same logical shape, and each output layout must be free of
/// zero-stride aliasing so every logical element has one mutable destination.
pub fn indexed_map4_inplace<A, B, C, D, F, const N: usize>(
    a: &mut ArrayViewMut<'_, A, N>,
    b: &mut ArrayViewMut<'_, B, N>,
    c: &mut ArrayViewMut<'_, C, N>,
    d: &mut ArrayViewMut<'_, D, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut([usize; N], &mut A, &mut B, &mut C, &mut D),
{
    if a.shape() != b.shape() || a.shape() != c.shape() || a.shape() != d.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }

    validate_mutable_output(a, "indexed multi-output map")?;
    validate_mutable_output(b, "indexed multi-output map")?;
    validate_mutable_output(c, "indexed multi-output map")?;
    validate_mutable_output(d, "indexed multi-output map")?;

    let size = a.layout().checked_size()?;
    let shape = a.shape();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let c_layout = c.layout();
    let d_layout = d.layout();
    let a_data = a.data_mut();
    let b_data = b.data_mut();
    let c_data = c.data_mut();
    let d_data = d.data_mut();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(());
    };
    let a_step = traversal.last_axis_stride(a_layout);
    let b_step = traversal.last_axis_stride(b_layout);
    let c_step = traversal.last_axis_stride(c_layout);
    let d_step = traversal.last_axis_stride(d_layout);
    for_each_row_major_indexed(
        traversal,
        |index| {
            Ok((
                a_layout.offset_of(index)? as isize,
                b_layout.offset_of(index)? as isize,
                c_layout.offset_of(index)? as isize,
                d_layout.offset_of(index)? as isize,
            ))
        },
        |index, offsets| {
            f(
                index,
                &mut a_data[offsets.0 as usize],
                &mut b_data[offsets.1 as usize],
                &mut c_data[offsets.2 as usize],
                &mut d_data[offsets.3 as usize],
            );
            offsets.0 += a_step;
            offsets.1 += b_step;
            offsets.2 += c_step;
            offsets.3 += d_step;
        },
    )?;

    Ok(())
}
