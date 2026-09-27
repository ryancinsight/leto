use super::immutable::ArrayView;
use super::mutable_core::ArrayViewMut;

/// Enable `view[[i, j, k]]` syntax, matching `leto::ArrayView` ergonomics.
impl<'a, T, const N: usize> std::ops::Index<[usize; N]> for ArrayView<'a, T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayView index out of bounds");
        &self.data[offset]
    }
}

/// Enable `view[[i, j, k]]` syntax on mutable views.
impl<'a, T, const N: usize> std::ops::Index<[usize; N]> for ArrayViewMut<'a, T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayViewMut index out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut index physical offset {offset} exceeds backing length {}",
            self.len
        );
        self.element(offset)
    }
}

/// Enable `view[[i, j, k]] = value` syntax on mutable views.
impl<'a, T, const N: usize> std::ops::IndexMut<[usize; N]> for ArrayViewMut<'a, T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayViewMut index_mut out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut index_mut physical offset {offset} exceeds backing length {}",
            self.len
        );
        self.element_mut(offset)
    }
}

/// Enable `view[i]` (usize) syntax for 1-D views (leto `ArrayView1` parity).
impl<'a, T> std::ops::Index<usize> for ArrayView<'a, T, 1> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayView1 index out of bounds");
        &self.data[offset]
    }
}

/// Enable `view[i]` (usize) mutable syntax for 1-D views.
impl<'a, T> std::ops::Index<usize> for ArrayViewMut<'a, T, 1> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayViewMut1 index out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut1 index physical offset {offset} exceeds backing length {}",
            self.len
        );
        self.element(offset)
    }
}

/// Enable `view[i] = value` syntax for 1-D mutable views.
impl<'a, T> std::ops::IndexMut<usize> for ArrayViewMut<'a, T, 1> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayViewMut1 index_mut out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut1 index_mut physical offset {offset} exceeds backing length {}",
            self.len
        );
        self.element_mut(offset)
    }
}
