use crate::domain::layout::Layout;
use crate::infrastructure::storage::Storage;
use serde::de::Error as DeserializeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::marker::PhantomData;

/// An N-dimensional strided array.
#[derive(Debug, Clone)]
pub struct Array<T, S, const N: usize> {
    pub(crate) layout: Layout<N>,
    pub(crate) storage: S,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, S, const N: usize> Serialize for Array<T, S, N>
where
    S: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> core::result::Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut state = serializer.serialize_struct("Array", 2)?;
        state.serialize_field("layout", &self.layout)?;
        state.serialize_field("storage", &self.storage)?;
        state.end()
    }
}

impl<'de, T, S, const N: usize> Deserialize<'de> for Array<T, S, N>
where
    S: Deserialize<'de> + Storage<T>,
{
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ArrayParts<S, const N: usize> {
            layout: Layout<N>,
            storage: S,
        }

        let parts = ArrayParts::<S, N>::deserialize(deserializer)?;
        Self::new(parts.layout, parts.storage).map_err(D::Error::custom)
    }
}
