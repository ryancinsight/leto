use crate::domain::error::{LetoError, Result};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// One leto-style slicing element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceArg {
    /// Select the full axis.
    All,
    /// Select a strided range on the current axis.
    Range {
        /// Inclusive logical start when present. Negative values count from the end.
        start: Option<isize>,
        /// Exclusive logical end when present. Negative values count from the end.
        end: Option<isize>,
        /// Non-zero stride. Negative strides reverse traversal.
        step: isize,
    },
    /// Select one element and remove the axis from the output rank.
    Index(isize),
    /// Insert a length-one axis with zero stride.
    NewAxis,
    /// Expand to as many full-axis selections as needed.
    Ellipsis,
}

impl SliceArg {
    /// Construct a strided range.
    #[inline]
    pub const fn range(start: Option<isize>, end: Option<isize>, step: isize) -> Self {
        Self::Range { start, end, step }
    }

    /// Construct an index selection.
    #[inline]
    pub const fn index(index: isize) -> Self {
        Self::Index(index)
    }

    /// Multiply the step size of this slice argument.
    #[inline]
    pub const fn step(self, step: isize) -> Self {
        match self {
            Self::Range {
                start,
                end,
                step: s,
            } => Self::Range {
                start,
                end,
                step: s * step,
            },
            // A full-axis selection carries an implicit step of 1; applying
            // `.step()` must yield a strided full range so that, e.g.,
            // `s![.., ..;-1, ..]` reverses the axis instead of silently
            // dropping the stride.
            Self::All => Self::Range {
                start: None,
                end: None,
                step,
            },
            other => other,
        }
    }
}

impl From<RangeFull> for SliceArg {
    #[inline]
    fn from(_: RangeFull) -> Self {
        Self::All
    }
}

impl From<Range<usize>> for SliceArg {
    #[inline]
    fn from(r: Range<usize>) -> Self {
        Self::Range {
            start: Some(r.start as isize),
            end: Some(r.end as isize),
            step: 1,
        }
    }
}

impl From<RangeTo<usize>> for SliceArg {
    #[inline]
    fn from(r: RangeTo<usize>) -> Self {
        Self::Range {
            start: None,
            end: Some(r.end as isize),
            step: 1,
        }
    }
}

impl From<RangeFrom<usize>> for SliceArg {
    #[inline]
    fn from(r: RangeFrom<usize>) -> Self {
        Self::Range {
            start: Some(r.start as isize),
            end: None,
            step: 1,
        }
    }
}

impl From<usize> for SliceArg {
    #[inline]
    fn from(index: usize) -> Self {
        Self::Index(index as isize)
    }
}

/// A normalized range selection for one input axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NormalizedRange {
    pub(crate) start: isize,
    pub(crate) len: usize,
    pub(crate) step: isize,
}

pub(crate) fn normalize_index(index: isize, axis_len: usize) -> Result<usize> {
    let len = isize::try_from(axis_len).map_err(|_| LetoError::Overflow {
        reason: "slice axis length conversion",
    })?;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Err(LetoError::OutOfBounds {
            index: vec![index.max(0) as usize],
            shape: vec![axis_len],
        });
    }
    Ok(normalized as usize)
}

pub(crate) fn normalize_range(
    start: Option<isize>,
    end: Option<isize>,
    step: isize,
    axis_len: usize,
) -> Result<NormalizedRange> {
    if step == 0 || step == isize::MIN {
        return Err(LetoError::IncompatibleSlice {
            range: (0, axis_len),
            shape: vec![axis_len],
        });
    }

    let len = isize::try_from(axis_len).map_err(|_| LetoError::Overflow {
        reason: "slice axis length conversion",
    })?;
    if step > 0 {
        let start = clamp_positive_bound(start.map_or(0, |value| resolve_bound(value, len)), len);
        let end = clamp_positive_bound(end.map_or(len, |value| resolve_bound(value, len)), len);
        let selected = if start >= end {
            0
        } else {
            ((end - start - 1) / step + 1) as usize
        };
        return Ok(NormalizedRange {
            start,
            len: selected,
            step,
        });
    }

    let reverse_step = -step;
    let start = clamp_negative_start(
        start.map_or(len - 1, |value| resolve_bound(value, len)),
        len,
    );
    let end = clamp_negative_end(end.map_or(-1, |value| resolve_bound(value, len)), len);
    let selected = if start <= end {
        0
    } else {
        ((start - end - 1) / reverse_step + 1) as usize
    };
    Ok(NormalizedRange {
        start,
        len: selected,
        step,
    })
}

#[inline]
fn resolve_bound(value: isize, len: isize) -> isize {
    if value < 0 {
        len + value
    } else {
        value
    }
}

#[inline]
fn clamp_positive_bound(value: isize, len: isize) -> isize {
    value.clamp(0, len)
}

#[inline]
fn clamp_negative_start(value: isize, len: isize) -> isize {
    if len == 0 {
        -1
    } else {
        value.clamp(-1, len - 1)
    }
}

#[inline]
fn clamp_negative_end(value: isize, len: isize) -> isize {
    if len == 0 {
        -1
    } else {
        value.clamp(-1, len - 1)
    }
}
