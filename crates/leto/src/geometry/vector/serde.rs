//! `serde` support for [`Vector`].
//!
//! Const-generic `[T; N]` is not `derive`-able, so `Vector` serializes as a
//! fixed tuple of `N` elements via these manual impls; the dependent geometry
//! types (`Point`, `Unit`, `Isometry3`, …) then derive over it.

use super::Vector;

impl<T: serde::Serialize, const N: usize> serde::Serialize for Vector<T, N> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(N)?;
        for x in &self.data {
            tup.serialize_element(x)?;
        }
        tup.end()
    }
}

impl<'de, T, const N: usize> serde::Deserialize<'de> for Vector<T, N>
where
    T: serde::Deserialize<'de> + Copy + Default,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Vis<T, const N: usize>(core::marker::PhantomData<T>);
        impl<'de, T, const N: usize> serde::de::Visitor<'de> for Vis<T, N>
        where
            T: serde::Deserialize<'de> + Copy + Default,
        {
            type Value = Vector<T, N>;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "a sequence of {N} numbers")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Vector<T, N>, A::Error> {
                let mut data = [T::default(); N];
                for (i, slot) in data.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(Vector { data })
            }
        }
        deserializer.deserialize_tuple(N, Vis(core::marker::PhantomData))
    }
}
