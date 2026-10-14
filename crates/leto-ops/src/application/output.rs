//! Static output representation for shared initialized/uninitialized kernels.

use std::mem::MaybeUninit;

pub(crate) trait OutputSlot<T: Copy>: Copy + Send + Sync {
    fn write(&mut self, value: T);

    /// The slot must have been written before this read.
    unsafe fn read(&self) -> T;

    #[inline]
    fn copy_from_slice(output: &mut [Self], input: &[T]) {
        assert_eq!(output.len(), input.len());
        for (slot, &value) in output.iter_mut().zip(input) {
            slot.write(value);
        }
    }

    #[inline]
    fn fill(output: &mut [Self], value: T) {
        for slot in output {
            slot.write(value);
        }
    }
}

impl<T: Copy + Send + Sync> OutputSlot<T> for T {
    #[inline]
    fn write(&mut self, value: T) {
        *self = value;
    }

    #[inline]
    unsafe fn read(&self) -> T {
        *self
    }

    #[inline]
    fn copy_from_slice(output: &mut [Self], input: &[T]) {
        output.copy_from_slice(input);
    }

    #[inline]
    fn fill(output: &mut [Self], value: T) {
        output.fill(value);
    }
}

impl<T: Copy + Send + Sync> OutputSlot<T> for MaybeUninit<T> {
    #[inline]
    fn write(&mut self, value: T) {
        MaybeUninit::write(self, value);
    }

    #[inline]
    unsafe fn read(&self) -> T {
        // SAFETY: the kernel must establish a preceding write for this slot.
        unsafe { self.assume_init() }
    }
}
