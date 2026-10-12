use leto::{Array, Layout, VecStorage};

pub(super) fn array<T: Clone, const R: usize>(
    shape: [usize; R],
    values: Vec<T>,
) -> Array<T, VecStorage<T>, R> {
    Array::new(
        Layout::c_contiguous(shape).expect("test shape is representable"),
        VecStorage::new(values),
    )
    .expect("test storage matches its shape")
}
