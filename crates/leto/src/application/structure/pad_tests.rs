use super::pad;
use crate::{ArrayView, Layout, LetoError, SliceArg, Storage};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[test]
fn padding_strings_preserves_strided_values_and_empty_shapes() {
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    let values = ["a", "unused", "b", "c", "unused", "d"].map(str::to_owned);
    let input = ArrayView::new(Layout::c_contiguous([2, 3]).expect("layout"), &values);
    let selected = input
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, 2)])
        .expect("strided input");
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    let output = pad(&selected, [(1, 0), (0, 1)], "pad".to_owned()).expect("padding");
    let empty = ArrayView::<String, 1>::new(Layout::c_contiguous([0]).expect("empty layout"), &[]);
    let empty_output = pad(&empty, [(0, 0)], "pad".to_owned()).expect("empty padding");
    let filled_empty = pad(&empty, [(1, 2)], "pad".to_owned()).expect("filled empty");
    drop(act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(output.shape(), [3, 3]);
    assert_eq!(
        output.storage().as_slice(),
        ["pad", "pad", "pad", "a", "b", "pad", "c", "d", "pad"]
    );
    assert_eq!(empty_output.shape(), [0]);
    assert_eq!(empty_output.storage().as_slice(), &[] as &[String]);
    assert_eq!(filled_empty.storage().as_slice(), ["pad", "pad", "pad"]);
}

#[derive(Debug)]
struct Payload {
    value: usize,
    drops: Rc<RefCell<Vec<usize>>>,
    clones: Rc<Cell<usize>>,
    panic_at: usize,
}

impl Clone for Payload {
    fn clone(&self) -> Self {
        let count = self.clones.get() + 1;
        self.clones.set(count);
        assert_ne!(count, self.panic_at, "clone panic at selected element");
        Self {
            value: self.value,
            drops: Rc::clone(&self.drops),
            clones: Rc::clone(&self.clones),
            panic_at: self.panic_at,
        }
    }
}

impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.borrow_mut().push(self.value);
    }
}

#[test]
fn padding_clone_panic_drops_only_initialized_prefix() {
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    let drops = Rc::new(RefCell::new(Vec::new()));
    let clones = Rc::new(Cell::new(0));
    let make = |value| Payload {
        value,
        drops: Rc::clone(&drops),
        clones: Rc::clone(&clones),
        panic_at: 3,
    };
    let source = [make(7), make(9)];
    let input = ArrayView::new(Layout::c_contiguous([2]).expect("layout"), &source);
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pad(&input, [(1, 1)], make(99))
    }))
    .expect_err("third clone must panic");
    drop(act);
    let _assert = tracing::info_span!("assert").entered();
    assert!(panic
        .downcast_ref::<String>()
        .expect("assertion panic text")
        .contains("clone panic at selected element"));
    assert_eq!(clones.get(), 3);
    assert_eq!(*drops.borrow(), [99, 7, 99]);
    assert_eq!(source.iter().map(|p| p.value).collect::<Vec<_>>(), [7, 9]);
    drop(source);
    assert_eq!(*drops.borrow(), [99, 7, 99, 7, 9]);
}

#[test]
fn padding_invalid_input_drops_initialized_prefix_and_returns_exact_error() {
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    let source = ["first".to_owned()];
    let invalid = ArrayView::new(Layout::c_contiguous([2]).expect("layout"), &source);
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    let error = pad(&invalid, [(1, 1)], "margin".to_owned()).expect_err("short storage");
    drop(act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(
        error,
        LetoError::StorageError {
            reason: "physical offset 1 exceeds backing slice length 1".to_owned()
        }
    );
    assert_eq!(source, ["first"]);
}
