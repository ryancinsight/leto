//! Fields holding only some x-planes of a grid.

use core::ops::Range;

use leto::{ArrayView3, ArrayViewMut3};

/// A field holding the x-planes `first..first + n` of a grid, `n` being the
/// view's first extent.
///
/// A chain of passes evaluated a slab of planes at a time keeps its
/// intermediates in a buffer the size of a slab rather than of the grid. The
/// buffer is reused for every slab, so it stays in cache, where a grid-sized
/// intermediate is written to planes last touched a whole pass earlier and
/// pays a read from DRAM for every line before it can write it. A window is
/// that buffer seen as the planes of the grid it holds: a derivative at grid
/// plane `x` reads the window's planes around `x` and takes the stencil the
/// grid's own extent gives `x`, not the window's.
#[derive(Clone, Copy)]
pub struct PlaneWindow<'a, T> {
    view: ArrayView3<'a, T>,
    first: usize,
}

impl<'a, T> PlaneWindow<'a, T> {
    /// `view` holding the grid planes from `first` on.
    #[must_use]
    pub fn new(view: ArrayView3<'a, T>, first: usize) -> Self {
        Self { view, first }
    }

    /// `view` holding every plane of its grid.
    #[must_use]
    pub fn whole(view: ArrayView3<'a, T>) -> Self {
        Self::new(view, 0)
    }

    /// The grid planes this window holds.
    #[must_use]
    pub fn planes(&self) -> Range<usize> {
        self.first..self.first + self.view.shape()[0]
    }

    pub(super) fn first(&self) -> usize {
        self.first
    }

    pub(super) fn shape(&self) -> [usize; 3] {
        self.view.shape()
    }
}

impl<'a, T: Copy> PlaneWindow<'a, T> {
    pub(super) fn view(&self) -> ArrayView3<'a, T> {
        self.view
    }
}

/// A destination holding the x-planes `first..first + n` of a grid; the
/// mutable counterpart of [`PlaneWindow`].
pub struct PlaneWindowMut<'view, 'a, T> {
    view: &'view mut ArrayViewMut3<'a, T>,
    first: usize,
}

impl<'view, 'a, T> PlaneWindowMut<'view, 'a, T> {
    /// `view` holding the grid planes from `first` on.
    #[must_use]
    pub fn new(view: &'view mut ArrayViewMut3<'a, T>, first: usize) -> Self {
        Self { view, first }
    }

    /// `view` holding every plane of its grid.
    #[must_use]
    pub fn whole(view: &'view mut ArrayViewMut3<'a, T>) -> Self {
        Self::new(view, 0)
    }

    /// The grid planes this window holds.
    #[must_use]
    pub fn planes(&self) -> Range<usize> {
        self.first..self.first + self.view.shape()[0]
    }

    pub(super) fn view_mut(&mut self) -> &mut ArrayViewMut3<'a, T> {
        self.view
    }

    pub(super) fn shape(&self) -> [usize; 3] {
        self.view.shape()
    }

    pub(super) fn first(&self) -> usize {
        self.first
    }
}
