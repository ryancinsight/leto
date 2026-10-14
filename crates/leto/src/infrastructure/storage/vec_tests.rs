use super::VecStorage;
use crate::{Storage, StorageMut};
use std::{cell::Cell, num::NonZeroU8, rc::Rc};

struct Payload {
    value: usize,
    drops: Rc<Cell<usize>>,
}

impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn uninitialized_slots_can_be_abandoned_without_dropping_payloads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct DiscardedZst;
    impl Drop for DiscardedZst {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    DROPS.store(0, Ordering::SeqCst);
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    let slots = VecStorage::<Payload>::uninit(3);
    assert_eq!(slots.as_slice().len(), 3);
    drop(slots);
    drop(VecStorage::<String>::uninit(1));
    drop(VecStorage::<NonZeroU8>::uninit(2));
    drop(VecStorage::<DiscardedZst>::uninit(3));
    drop(act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(DROPS.load(Ordering::SeqCst), 0);
}

#[test]
fn initialized_slots_preserve_allocation_values_and_destructor_counts() {
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    let drops = Rc::new(Cell::new(0));
    let expected = [2, 7, 19];
    let mut slots = VecStorage::<Payload>::uninit(expected.len());
    let pointer = slots.as_slice().as_ptr().cast::<Payload>();
    for (slot, value) in slots.as_mut_slice().iter_mut().zip(expected) {
        slot.write(Payload {
            value,
            drops: Rc::clone(&drops),
        });
    }
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    // SAFETY: every slot was written with a uniquely owned Payload above.
    let initialized = unsafe { slots.assume_init() };
    drop(act);
    let assertion = tracing::info_span!("assert").entered();
    assert_eq!(initialized.as_slice().as_ptr(), pointer);
    assert_eq!(
        initialized
            .as_slice()
            .iter()
            .map(|p| p.value)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(drops.get(), 0);
    drop(initialized);
    assert_eq!(drops.get(), expected.len());
    drop(assertion);
}

#[test]
fn strings_and_nonzero_values_are_initialized_before_typed_access() {
    let _trace = crate::test_trace::capture();
    let arrange = tracing::info_span!("arrange").entered();
    let mut strings = VecStorage::<String>::uninit(2);
    for (slot, value) in strings.as_mut_slice().iter_mut().zip(["alpha", "omega"]) {
        slot.write(value.to_owned());
    }
    let mut numbers = VecStorage::<NonZeroU8>::uninit(3);
    for (slot, value) in numbers.as_mut_slice().iter_mut().zip([3, 5, 11]) {
        slot.write(NonZeroU8::new(value).expect("nonzero test value"));
    }
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    // SAFETY: all slots contain the valid values written above.
    let strings = unsafe { strings.assume_init() };
    let numbers = unsafe { numbers.assume_init() };
    drop(act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(strings.as_slice(), ["alpha", "omega"]);
    assert_eq!(
        numbers
            .as_slice()
            .iter()
            .map(|n| n.get())
            .collect::<Vec<_>>(),
        [3, 5, 11]
    );
}

#[test]
fn empty_and_overaligned_slots_preserve_typed_alignment() {
    #[repr(align(8192))]
    struct Aligned(u8);
    let _trace = crate::test_trace::capture();
    for len in [0, 1, 3] {
        let arrange = tracing::info_span!("arrange", len).entered();
        let mut slots = VecStorage::<Aligned>::uninit(len);
        let pointer = slots.as_slice().as_ptr().cast::<Aligned>();
        for slot in slots.as_mut_slice() {
            slot.write(Aligned(17));
        }
        drop(arrange);
        let act = tracing::info_span!("act").entered();
        // SAFETY: every slot is written, or len is zero.
        let values = unsafe { slots.assume_init() };
        drop(act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(pointer.addr() % std::mem::align_of::<Aligned>(), 0);
        assert_eq!(values.as_slice().as_ptr(), pointer);
        assert_eq!(
            values.as_slice().iter().map(|v| v.0).collect::<Vec<_>>(),
            vec![17; len]
        );
    }
}

#[test]
fn initialized_zero_sized_payloads_drop_once_per_slot() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Zst;
    impl Drop for Zst {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }
    let _trace = crate::test_trace::capture();
    for len in [0, 1, 5] {
        let arrange = tracing::info_span!("arrange", len).entered();
        DROPS.store(0, Ordering::SeqCst);
        let mut slots = VecStorage::<Zst>::uninit(len);
        for slot in slots.as_mut_slice() {
            slot.write(Zst);
        }
        drop(arrange);
        let act = tracing::info_span!("act").entered();
        // SAFETY: all len zero-sized values were initialized above.
        let values = unsafe { slots.assume_init() };
        assert_eq!(values.len(), len);
        drop(values);
        drop(act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(DROPS.load(Ordering::SeqCst), len);
    }
}
