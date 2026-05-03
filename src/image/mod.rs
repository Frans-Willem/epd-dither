//! `image`-Cargo-feature group: glue between this crate's abstract
//! image-shaped traits ([`crate::dither::image_traits`]) and the
//! [`image`](https://docs.rs/image) crate, plus a palette-indexed PNG sink.

pub mod adapter;
pub mod palette_image;
