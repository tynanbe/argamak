import argamak/util

/// Numerical formats for tensors.
///
/// Each `Format` uses a specific number of bits to represent a data point.
///
pub opaque type Format(a) {
  // Brain Floating Point
  Bfloat16
  // Floating Point
  Float16
  Float32
  Float64
  // Signed Integer
  Int8
  Int16
  Int32
  Int64
  // Unsigned Integer
  Uint8
  Uint16
  Uint32
  Uint64
}

/// Creates a 32-bit floating-point `Format`.
///
pub fn float32() -> Format(Float) {
  Float32
}

/// Creates a 32-bit signed integer `Format`.
///
pub fn int32() -> Format(Int) {
  Int32
}

if erlang {
  /// Creates a 64-bit floating-point `Format`.
  ///
  pub fn float64() -> Format(Float) {
    Float64
  }

  /// Creates a 64-bit signed integer `Format`.
  ///
  pub fn int64() -> Format(Int) {
    Int64
  }

  /// Creates a 64-bit unsigned integer `Format`.
  ///
  pub fn uint64() -> Format(Int) {
    Uint64
  }

  /// Creates a 32-bit unsigned integer `Format`.
  ///
  pub fn uint32() -> Format(Int) {
    Uint32
  }

  /// Creates a 16-bit brain floating-point format.
  ///
  pub fn bfloat16() -> Format(Float) {
    Bfloat16
  }

  /// Creates a 16-bit floating-point `Format`.
  ///
  pub fn float16() -> Format(Float) {
    Float16
  }

  /// Creates a 16-bit signed integer `Format`.
  ///
  pub fn int16() -> Format(Int) {
    Int16
  }

  /// Creates a 16-bit unsigned integer `Format`.
  ///
  pub fn uint16() -> Format(Int) {
    Uint16
  }

  /// Creates an 8-bit signed integer `Format`.
  ///
  pub fn int8() -> Format(Int) {
    Int8
  }

  /// Creates an 8-bit unsigned integer `Format`.
  ///
  pub fn uint8() -> Format(Int) {
    Uint8
  }
}

/// Converts a given `Format` into its native representation.
///
pub fn to_native(format: Format(a)) -> Native {
  do_to_native(format)
}

if erlang {
  /// A type for `Native` format representations.
  ///
  pub opaque type Native {
    Bf(size: Int)
    F(size: Int)
    S(size: Int)
    U(size: Int)
  }

  fn do_to_native(format: Format(a)) -> Native {
    case format {
      Float64 -> F(64)
      Int64 -> S(64)
      Uint64 -> U(64)
      Float32 -> F(32)
      Int32 -> S(32)
      Uint32 -> U(32)
      Bfloat16 -> Bf(16)
      Float16 -> F(16)
      Int16 -> S(16)
      Uint16 -> U(16)
      Int8 -> S(8)
      Uint8 -> U(8)
    }
  }
}

if javascript {
  /// TODO
  ///
  pub opaque type Native {
    Native(size: Int)
  }

  fn do_to_native(format: Format(a)) -> Native {
    todo
  }
}

/// Converts a `Format` into a `String`.
///
pub fn to_string(format: Format(a)) -> String {
  assert Ok(string) = util.record_to_string(format)
  string
}
