import gleam/string

/// Numerical formats for tensors.
///
/// Each `Format` uses a set number of bits to represent every `Float`-like or
/// `Int`-like value.
///
pub opaque type Format(a) {
  Format(a)
}

/// A 32-bit floating point type, argamak's standard for working with floats.
///
pub type Float32 {
  Float32
}

/// Creates a 32-bit floating point `Format`, argamak's standard for working
/// with floats.
///
pub fn float32() -> Format(Float32) {
  Format(Float32)
}

/// A 32-bit signed integer type, argamak's standard for working with ints.
///
pub type Int32 {
  Int32
}

/// Creates a 32-bit signed integer `Format`, argamak's standard for working
/// with ints.
///
pub fn int32() -> Format(Int32) {
  Format(Int32)
}

@target(erlang)
/// A 64-bit floating point type.
///
pub type Float64 {
  Float64
}

@target(erlang)
/// Creates a 64-bit floating point `Format`.
///
pub fn float64() -> Format(Float64) {
  Format(Float64)
}

@target(erlang)
/// A 64-bit signed integer type.
///
pub type Int64 {
  Int64
}

@target(erlang)
/// Creates a 64-bit signed integer `Format`.
///
pub fn int64() -> Format(Int64) {
  Format(Int64)
}

@target(erlang)
/// A 64-bit unsigned integer type.
///
pub type Uint64 {
  Uint64
}

@target(erlang)
/// Creates a 64-bit unsigned integer `Format`.
///
pub fn uint64() -> Format(Uint64) {
  Format(Uint64)
}

@target(erlang)
/// A 32-bit unsigned integer type.
///
pub type Uint32 {
  Uint32
}

@target(erlang)
/// Creates a 32-bit unsigned integer `Format`.
///
pub fn uint32() -> Format(Uint32) {
  Format(Uint32)
}

@target(erlang)
/// A 16-bit brain floating point type.
///
pub type Bfloat16 {
  Bfloat16
}

@target(erlang)
/// Creates a 16-bit brain floating point `Format`.
///
pub fn bfloat16() -> Format(Bfloat16) {
  Format(Bfloat16)
}

@target(erlang)
/// A 16-bit floating point type.
///
pub type Float16 {
  Float16
}

@target(erlang)
/// Creates a 16-bit floating point `Format`.
///
pub fn float16() -> Format(Float16) {
  Format(Float16)
}

@target(erlang)
/// A 16-bit signed integer type.
///
pub type Int16 {
  Int16
}

@target(erlang)
/// Creates a 16-bit signed integer `Format`.
///
pub fn int16() -> Format(Int16) {
  Format(Int16)
}

@target(erlang)
/// A 16-bit unsigned integer type.
///
pub type Uint16 {
  Uint16
}

@target(erlang)
/// Creates a 16-bit unsigned integer `Format`.
///
pub fn uint16() -> Format(Uint16) {
  Format(Uint16)
}

@target(erlang)
/// An 8-bit signed integer type.
///
pub type Int8 {
  Int8
}

@target(erlang)
/// Creates an 8-bit signed integer `Format`.
///
pub fn int8() -> Format(Int8) {
  Format(Int8)
}

@target(erlang)
/// An 8-bit unsigned integer type.
///
pub type Uint8 {
  Uint8
}

@target(erlang)
/// Creates an 8-bit unsigned integer `Format`.
///
pub fn uint8() -> Format(Uint8) {
  Format(Uint8)
}

/// A type for `Native` format representations.
///
pub type Native

/// Converts a given `Format` into its native representation.
///
pub fn to_native(format: Format(a)) -> Native {
  let Format(x) = format
  do_to_native(x)
}

@external(erlang, "argamak_ffi", "format_to_native")
@external(javascript, "../argamak_ffi.mjs", "format_to_native")
fn do_to_native(format: a) -> Native

/// Converts a `Format` into a `String`.
///
pub fn to_string(format: Format(a)) -> String {
  string.inspect(format)
}
