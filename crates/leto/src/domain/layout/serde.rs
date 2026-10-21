//! `Serialize`/`Deserialize` for [`Layout`].

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::core::Layout;

impl<const N: usize> Serialize for Layout<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Layout", 3)?;
        state.serialize_field("shape", self.shape.as_slice())?;
        state.serialize_field("strides", self.strides.as_slice())?;
        state.serialize_field("offset", &self.offset)?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for Layout<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LayoutFields {
            shape: Vec<usize>,
            strides: Vec<isize>,
            offset: usize,
        }

        let fields = LayoutFields::deserialize(deserializer)?;
        let shape = fields.shape.try_into().map_err(|values: Vec<usize>| {
            serde::de::Error::custom(format!(
                "layout shape rank {} does not match expected rank {}",
                values.len(),
                N
            ))
        })?;
        let strides = fields.strides.try_into().map_err(|values: Vec<isize>| {
            serde::de::Error::custom(format!(
                "layout stride rank {} does not match expected rank {}",
                values.len(),
                N
            ))
        })?;

        // Deserialization is construction: route it through the validating
        // path so a hostile or corrupt payload cannot mint a layout that safe
        // code could not have built.
        Self::try_new(shape, strides, fields.offset).map_err(serde::de::Error::custom)
    }
}
