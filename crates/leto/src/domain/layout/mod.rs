mod broadcast;
mod contiguity;
pub(crate) mod kernels;
mod shape;
mod slice_with;
mod strides;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Represents an N-dimensional strided layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout<const N: usize> {
    /// The shape of the array (size of each dimension).
    pub shape: [usize; N],
    /// The stride of each dimension in elements (not bytes).
    pub strides: [isize; N],
    /// The starting offset in the storage buffer.
    pub offset: usize,
}

impl<const N: usize> Layout<N> {
    /// Create a new layout with explicit shape, strides, and offset.
    pub const fn new(shape: [usize; N], strides: [isize; N], offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }
}

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

        Ok(Self {
            shape,
            strides,
            offset: fields.offset,
        })
    }
}
