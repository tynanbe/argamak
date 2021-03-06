import argamak/format.{Format}
import argamak/space.{D0, Space}
import gleam/dynamic.{DecodeError, DecodeErrors, Dynamic}
import gleam/float
import gleam/function
import gleam/int
import gleam/io
import gleam/iterator
import gleam/list
import gleam/option.{None, Option, Some}
import gleam/pair
import gleam/regex
import gleam/result
import gleam/string

if erlang {
  import gleam/erlang.{Crash}
}

/// A `Tensor` is a generic container for n-dimensional data structures.
///
pub opaque type Tensor(a, dn, axis) {
  Tensor(data: Native, format: Format(a), space: Space(dn, axis))
}

/// A type for `Native` tensor representations.
///
pub external type Native

/// When a tensor operation cannot succeed.
///
pub type TensorError {
  EmptyTensor
  IncompatibleAxes
  IncompatibleShape
  InvalidData
  SpaceErrors(space.SpaceErrors)
}

/// Converts an error message string into a `TensorError`.
///
fn error(message: String) -> TensorError {
  do_error(message)
}

if erlang {
  fn do_error(message: String) -> TensorError {
    [
      #(EmptyTensor, "empty tensor"),
      #(IncompatibleAxes, "axes .*? must be unique integers"),
      #(IncompatibleAxes, "broadcast axes must be ordered"),
      #(IncompatibleAxes, "cannot merge names"),
      #(IncompatibleShape, "cannot broadcast"),
      #(IncompatibleShape, "cannot reshape"),
      #(IncompatibleShape, "invalid dimension"),
      #(InvalidData, "cannot infer the numerical type"),
    ]
    |> replace_error(when: message, found_in: _)
  }
}

/// Converts a `Float` into a `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > let tensor = from_float(1.)
/// > print(from: tensor, wrap_at: -1, meta: False)
/// // 1.0
/// Nil
/// ```
///
pub fn from_float(float: Float) -> Tensor(Float, D0, Nil) {
  assert Ok(space) = space.d0()
  assert Ok(tensor) = tensor(from: float, into: space, with: format.float32)
  tensor
}

/// Converts an `Int` into a `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > let tensor = from_int(1)
/// > print(from: tensor, wrap_at: -1, meta: False)
/// // 1
/// Nil
/// ```
///
pub fn from_int(int: Int) -> Tensor(Int, D0, Nil) {
  assert Ok(space) = space.d0()
  assert Ok(tensor) = tensor(from: int, into: space, with: format.int32)
  tensor
}

/// Results in a `Tensor` created from a list of floats and placed into a given
/// n-dimensional `Space` on success, or a `TensorError` on failure.
///
/// The `Space`'s `Shape` may have a single dimension given as `-1`, in which
/// case that dimension's size will be inferred from the given list. This is
/// useful when working with lists of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1.], into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D1 #(X, 1)
/// // data:
/// // [1.0]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, 2))
/// > try tensor = from_floats(of: [1., 2., 3., 4.], into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D2 #(X, 2), #(Y, 2)
/// // data:
/// // [[1.0, 2.0],
/// //  [3.0, 4.0]]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d3(#(X, -1), #(Y, 2), #(Z, 2))
/// > let list = [1., 2., 3., 4., 5., 6., 7., 8.]
/// > try tensor = from_floats(of: list, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D3 #(X, 2), #(Y, 2), #(Z, 2)
/// // data:
/// // [[[1.0, 2.0],
/// //   [3.0, 4.0]],
/// //  [[5.0, 6.0],
/// //   [7.0, 8.0]]]
/// Ok(Nil)
/// ```
///
pub fn from_floats(
  of list: List(Float),
  into space: Space(dn, axis),
) -> Result(Tensor(Float, dn, axis), TensorError) {
  tensor(from: list, into: space, with: format.float32)
}

/// Results in a `Tensor` created from a list of integers and placed into a
/// given n-dimensional `Space` on success, or a `TensorError` on failure.
///
/// The `Space`'s `Shape` may have a single dimension given as `-1`, in which
/// case that dimension's size will be inferred from the given list. This is
/// useful when working with lists of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_ints(of: [1], into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D1 #(X, 1)
/// // data:
/// // [1]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, 2))
/// > try tensor = from_ints(of: [1, 2, 3, 4], into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D2 #(X, 2), #(Y, 2)
/// // data:
/// // [[1, 2],
/// //  [3, 4]]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d3(#(X, -1), #(Y, 2), #(Z, 2))
/// > let list = [1, 2, 3, 4, 5, 6, 7, 8]
/// > try tensor = from_ints(of: list, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D3 #(X, 2), #(Y, 2), #(Z, 2)
/// // data:
/// // [[[1, 2],
/// //   [3, 4]],
/// //  [[5, 6],
/// //   [7, 8]]]
/// Ok(Nil)
/// ```
///
pub fn from_ints(
  of list: List(Int),
  into space: Space(dn, axis),
) -> Result(Tensor(Int, dn, axis), TensorError) {
  tensor(from: list, into: space, with: format.int32)
}

/// Results in a `Tensor` created from a `Native` representation on success, or
/// a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/format
/// > import argamak/space
/// > import gleam/dynamic.{Dynamic}
/// > external fn erlang_tensor(Dynamic) -> Native =
/// >   "Elixir.Nx" "tensor"
/// > let native = erlang_tensor(dynamic.from([[1, 2], [3, 4]]))
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, -1))
/// > try tensor = from_native(of: native, into: space, with: format.int32)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D2 #(X, 2), #(Y, 2)
/// // data:
/// // [[1, 2],
/// //  [3, 4]]
/// Ok(Nil)
/// ```
///
pub fn from_native(
  of native: Native,
  into space: Space(dn, axis),
  with format: fn() -> Format(a),
) -> Result(Tensor(a, dn, axis), TensorError) {
  tensor(from: do_to_list(native), into: space, with: format)
}

/// Results in a `Tensor` put into a given `Space` with a `Format` applied to
/// the data on success, or a `TensorError` on failure.
///
fn tensor(
  from data: a,
  into space: Space(dn, axis),
  with format: fn() -> Format(b),
) -> Result(Tensor(b, dn, axis), TensorError) {
  fn() { do_tensor(data, space, format) }
  |> rescue(apply: error)
  |> result.flatten
}

if erlang {
  fn do_tensor(
    data: a,
    space: Space(dn, axis),
    format: fn() -> Format(b),
  ) -> Result(Tensor(b, dn, axis), TensorError) {
    Tensor(data: erlang_tensor(data, []), format: format(), space: space)
    |> reshape(into: space)
    |> result.map(with: as_format(for: _, apply: format))
  }

  type Opt(axis) {
    Axes(List(Int))
    Names(List(axis))
  }

  external fn erlang_tensor(a, List(Opt(axis))) -> Native =
    "Elixir.Nx" "tensor"
}

/// Returns the axes of a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > axes(from_int(3))
/// []
///
/// > import argamak/space
/// > type Axis { X }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1., 2., 3.], into: space)
/// > Ok(axes(tensor))
/// Ok([X])
///
/// > type OtherAxis { Alpha Omega }
/// > assert Ok(space) = space.d2(#(Alpha, 1), #(Omega, 3))
/// > try tensor = from_ints(of: [1, 2, 3], into: space)
/// > Ok(axes(tensor))
/// Ok([Alpha, Omega])
/// ```
///
pub fn axes(tensor: Tensor(a, dn, axis)) -> List(axis) {
  tensor
  |> space
  |> space.axes
}

/// Returns the `Format` of a given `Tensor`.
///
/// See `argamak/format` for more information.
///
/// ## Examples
///
/// ```gleam
/// import argamak/format
/// > format(from_float(0.))
/// format.float32()
///
/// > import argamak/space
/// > type Axis { X }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_ints(of: [1, 2, 3], into: space)
/// > Ok(format(tensor))
/// Ok(format.int32())
/// ```
///
pub fn format(tensor: Tensor(a, dn, axis)) -> Format(a) {
  tensor.format
}

/// Returns the rank of a given `Tensor` as an `Int` representing the number of
/// dimensions.
///
/// ## Examples
///
/// ```gleam
/// > rank(from_float(0.))
/// 0
///
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_ints(of: [1, 2, 3], into: space)
/// > Ok(rank(tensor))
/// Ok(1)
///
/// > assert Ok(space) = space.d3(#(X, 2), #(Y, 2), #(Z, 2))
/// > try tensor = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: space)
/// > Ok(rank(tensor))
/// Ok(3)
/// ```
///
pub fn rank(tensor: Tensor(a, dn, axis)) -> Int {
  tensor
  |> space
  |> space.elements
  |> list.length
}

/// Returns the shape of a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > shape(from_float(0.))
/// []
///
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_ints(of: [1, 2, 3], into: space)
/// > Ok(shape(tensor))
/// Ok([3])
///
/// > assert Ok(space) = space.d3(#(X, 2), #(Y, 2), #(Z, 2))
/// > try tensor = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: space)
/// > Ok(shape(tensor))
/// Ok([2, 2, 2])
/// ```
///
pub fn shape(tensor: Tensor(a, dn, axis)) -> List(Int) {
  tensor
  |> space
  |> space.shape
}

/// Returns the `Space` a given `Tensor` is currently in.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > assert Ok(space) = space.d0()
/// > space(from_float(0.))
/// space
///
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_ints(of: [1, 2, 3], into: space)
/// > Ok(space(tensor))
/// Ok(space)
///
/// > assert Ok(space) = space.d3(#(X, 2), #(Y, 2), #(Z, 2))
/// > try tensor = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: space)
/// > Ok(space(tensor))
/// Ok(space)
/// ```
///
pub fn space(tensor: Tensor(a, dn, axis)) -> Space(dn, axis) {
  tensor.space
}

/// Changes the `Format` of a `Tensor`.
///
/// Converting from float formats to integer formats truncates the data. For
/// consistency, consider using `round`, `floor`, or `ceil` before casting from
/// float formats to integer formats.
///
/// Lowering precision may lead to an overflow or underflow, the outcome of
/// which depends on platform and compiler.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/format
/// > as_format(for: from_int(0), apply: format.float32)
/// from_float(0.)
///
/// > import argamak/space
/// > type Axis { X }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1., 2., 3.], into: space)
/// > Ok(as_format(for: tensor, apply: format.int32))
/// from_ints(of: [1, 2, 3], into: space)
/// ```
///
pub fn as_format(
  for tensor: Tensor(a, dn, axis),
  apply format: fn() -> Format(b),
) -> Tensor(b, dn, axis) {
  Tensor(
    data: format()
    |> format.to_native
    |> do_as_format(to_native(tensor), _),
    format: format(),
    space: space(tensor),
  )
}

if erlang {
  external fn do_as_format(Native, format.Native) -> Native =
    "Elixir.Nx" "as_type"
}

/// Results in a `Tensor` broadcast into a given n-dimensional `Space` on
/// success, or a `TensorError` on failure.
///
/// The new `Space` cannot be of lower dimensionality than the `Tensor`'s
/// current `Space`.
///
/// The new `Tensor`'s current `Space` must have dimension sizes compatible with
/// the right-most dimensions of the new `Space`; that is, every current
/// dimension size must be `1` or equal to its counterpart in the new `Space`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > let tensor = from_int(0)
/// > assert Ok(space) = space.d1(#(X, 3))
/// > try tensor = broadcast(from: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D1 #(X, 3)
/// // data:
/// // [0, 0, 0]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d1(#(X, -1))
/// > let tensor = from_ints(of: [-1], into: space)
/// > assert Ok(space) = space.d1(#(Y, 5))
/// > try tensor = broadcast(from: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D1 #(Y, 5)
/// // data:
/// // [-1, -1, -1, -1, -1]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1., 2., 3.], into: space)
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, 3))
/// > try tensor = broadcast(from: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D2 #(X, 2), #(Y, 3)
/// // data:
/// // [[1.0, 2.0, 3.0],
/// //  [1.0, 2.0, 3.0]]
/// Ok(Nil)
/// ```
///
pub fn broadcast(
  from tensor: Tensor(a, b, c),
  into space: Space(dn, axis),
) -> Result(Tensor(a, dn, axis), TensorError) {
  fn() { do_broadcast(tensor, space) }
  |> rescue(apply: error)
}

if erlang {
  fn do_broadcast(
    tensor: Tensor(a, b, c),
    space: Space(dn, axis),
  ) -> Tensor(a, dn, axis) {
    let shape =
      space
      |> space.shape
      |> list_to_tuple
    let opt =
      space
      |> space.axes
      |> Names

    Tensor(
      data: tensor
      |> to_native
      |> erlang_broadcast(shape, [opt]),
      format: format(tensor),
      space: space,
    )
  }

  external fn erlang_broadcast(Native, tuple, List(Opt(axis))) -> Native =
    "Elixir.Nx" "broadcast"
}

/// Results in a `Tensor` broadcast into a given n-dimensional `Space` on
/// success, or a `TensorError` on failure.
///
/// The new `Space` cannot be of lower dimensionality than the `Tensor`'s
/// current `Space`.
///
/// The given function will be used to map from the elements of the `Tensor`'s
/// current `Space` to axes of the new `Space`, allowing broadcasting into a
/// `Space` that would be incompatible for a standard `broadcast` operation.
/// The function must uniquely map the current `Space`'s axes to the new
/// `Space`'s axes; that is, duplicate axes will result in a `TensorError`.
/// Furthermore, the current `Space`'s axes must maintain the same relative
/// order when mapped to the new `Space`.
///
/// The new `Tensor`'s current `Space` must have dimension sizes compatible with
/// the matching dimensions of the new `Space`; that is, every current dimension
/// size must be `1` or equal to its counterpart in the new `Space`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1., 2., 3.], into: space)
/// > assert Ok(space) = space.d2(#(X, 3), #(Y, 2))
/// > try new_tensor = broadcast_over(
///     from: tensor,
///     into: space,
///     with: fn(_) { X },
///   )
/// > Ok(print(from: new_tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D2 #(X, 3), #(Y, 2)
/// // data:
/// // [[1.0, 1.0],
/// //  [2.0, 2.0],
/// //  [3.0, 3.0]]
/// Ok(Nil)
///
/// > try tensor = from_ints(of: [1, 2, 3, 4, 5, 6], into: space)
/// > assert Ok(space) = space.d3(#(X, 3), #(Y, 2), #(Z, 2))
/// > try new_tensor = broadcast_over(
///     from: tensor,
///     into: space,
///     with: fn(element) {
///       let #(axis, _size) = element
///       case axis {
///         X -> X
///         Y -> Z
///       }
///     },
///   )
/// > Ok(print(from: new_tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D3 #(X, 3), #(Y, 2), #(Z, 2)
/// // data:
/// // [[[1, 2],
/// //   [1, 2]],
/// //  [[3, 4],
/// //   [3, 4]],
/// //  [[5, 6],
/// //   [5, 6]]]
/// Ok(Nil)
///
/// > import gleam/pair
/// > try new_tensor = broadcast_over(
///     from: tensor,
///     into: space,
///     with: pair.first,
///   )
/// > Ok(print(from: new_tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Int32
/// // space: D3 #(X, 3), #(Y, 2), #(Z, 2)
/// // data:
/// // [[[1, 1],
/// //   [2, 2]],
/// //  [[3, 3],
/// //   [4, 4]],
/// //  [[5, 5],
/// //   [6, 6]]]
/// Ok(Nil)
/// ```
///
pub fn broadcast_over(
  from tensor: Tensor(a, b, c),
  into space: Space(dn, axis),
  with axes: fn(#(c, Int)) -> axis,
) -> Result(Tensor(a, dn, axis), TensorError) {
  let indexed_axes =
    space
    |> space.axes
    |> iterator.from_list
    |> iterator.index
  try axes =
    tensor.space
    |> space.elements
    |> list.map(with: function.compose(
      axes,
      fn(axis) {
        indexed_axes
        |> iterator.find(one_that: function.compose(
          pair.second,
          fn(new_axis) { new_axis == axis },
        ))
        |> result.map(with: pair.first)
        |> result.replace_error(IncompatibleAxes)
      },
    ))
    |> result.all

  fn() { do_broadcast_over(tensor, space, axes) }
  |> rescue(apply: error)
}

if erlang {
  fn do_broadcast_over(
    tensor: Tensor(a, b, c),
    space: Space(dn, axis),
    axes: List(Int),
  ) -> Tensor(a, dn, axis) {
    let shape =
      space
      |> space.shape
      |> list_to_tuple
    let opts = [
      Axes(axes),
      space
      |> space.axes
      |> Names,
    ]

    Tensor(
      data: tensor
      |> to_native
      |> erlang_broadcast(shape, opts),
      format: format(tensor),
      space: space,
    )
  }
}

/// Results in a `Tensor` placed into a given n-dimensional `Space` on success,
/// or a `TensorError` on failure.
///
/// The `Space`'s `Shape` may have a single dimension given as `-1`, in which
/// case that dimension's size will be inferred from the given list. This is
/// useful when working with lists of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > let tensor = from_float(1.)
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = reshape(put: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D1 #(X, 1)
/// // data:
/// // [1.0]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d1(#(X, -1))
/// > try tensor = from_floats(of: [1., 2., 3., 4.], into: space)
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, 2))
/// > try tensor = reshape(put: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D2 #(X, 2), #(Y, 2)
/// // data:
/// // [[1.0, 2.0],
/// //  [3.0, 4.0]]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, -1))
/// > let list = [1., 2., 3., 4., 5., 6., 7., 8.]
/// > try tensor = from_floats(of: list, into: space)
/// > assert Ok(space) = space.d3(#(X, -1), #(Y, 2), #(Z, 2))
/// > try tensor = reshape(put: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: True))
/// // Tensor
/// // format: Float32
/// // space: D3 #(X, 2), #(Y, 2), #(Z, 2)
/// // data:
/// // [[[1.0, 2.0],
/// //   [3.0, 4.0]],
/// //  [[5.0, 6.0],
/// //   [7.0, 8.0]]]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d3(#(X, 1), #(Y, 1), #(Z, 1))
/// > try tensor = from_floats(of: [1.], into: space)
/// > assert Ok(space) = space.d0()
/// > try tensor = reshape(put: tensor, into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: False))
/// // 1.0
/// Ok(Nil)
/// ```
///
pub fn reshape(
  put tensor: Tensor(a, b, c),
  into space: Space(dn, axis),
) -> Result(Tensor(a, dn, axis), TensorError) {
  try tensor =
    Tensor(data: to_native(tensor), format: format(tensor), space: space)
    |> fit

  fn() { do_reshape(tensor) }
  |> rescue(apply: error)
}

if erlang {
  fn do_reshape(tensor: Tensor(a, dn, axis)) -> Tensor(a, dn, axis) {
    let space = space(tensor)
    let shape =
      space
      |> space.shape
      |> list_to_tuple
    let opt =
      space
      |> space.axes
      |> Names
    map_data(over: tensor, with: erlang_reshape(_, shape, [opt]))
  }

  external fn erlang_reshape(Native, tuple, List(Opt(axis))) -> Native =
    "Elixir.Nx" "reshape"
}

/// Converts a `Tensor` without dimensions into a `Float`.
///
/// ## Examples
///
/// ```gleam
/// > let tensor = from_float(0.)
/// > to_float(tensor)
/// 0.
///
/// > import argamak/format
/// > let tensor = from_int(0)
/// > let tensor = as_format(for: tensor, apply: format.float32)
/// > to_float(tensor)
/// 0.
/// ```
///
pub fn to_float(tensor: Tensor(Float, D0, axis)) -> Float {
  tensor
  |> to_native
  |> to_number
}

/// Converts a `Tensor` without dimensions into an `Int`.
///
/// ## Examples
///
/// ```gleam
/// > let tensor = from_int(0)
/// > to_int(tensor)
/// 0
///
/// > import argamak/format
/// > let tensor = from_float(0.)
/// > let tensor = as_format(for: tensor, apply: format.int32)
/// > to_int(tensor)
/// 0
/// ```
///
pub fn to_int(tensor: Tensor(Int, D0, axis)) -> Int {
  tensor
  |> to_native
  |> to_number
}

if erlang {
  external fn to_number(Native) -> a =
    "Elixir.Nx" "to_number"
}

/// Converts a `Tensor` into a flat list of numbers.
///
/// ## Examples
///
/// ```gleam
/// > to_list(from_int(0))
/// [0]
///
/// > import argamak/space
/// > type Axis { X Y }
/// > assert Ok(space) = space.d2(#(X, 3), #(Y, 1))
/// > try tensor = from_floats(of: [1., 2., 3.], into: space)
/// > Ok(to_list(tensor))
/// Ok([1., 2., 3.])
/// ```
///
pub fn to_list(tensor: Tensor(a, dn, axis)) -> List(a) {
  tensor
  |> to_native
  |> do_to_list
}

if erlang {
  external fn do_to_list(Native) -> List(a) =
    "Elixir.Nx" "to_flat_list"
}

/// Coverts a `Tensor` into its `Native` representation.
///
/// ## Examples
///
/// ```gleam
/// > external fn erlang_rank(Native) -> Int =
/// >   "Elixir.Nx" "rank"
/// > let native = to_native(from_int(3))
/// > erlang_rank(native)
/// 0
/// ```
///
pub fn to_native(tensor: Tensor(a, dn, axis)) -> Native {
  tensor.data
}

/// Results in the given `Tensor` on success, or a `TensorError` on failure.
///
/// Ensures the `Tensor` is compatible with its `Space` and computes up to one
/// inferred dimension size.
///
/// Useful within low-level functions in which a `Tensor` is put into a new
/// `Space`.
///
fn fit(tensor: Tensor(a, dn, axis)) -> Result(Tensor(a, dn, axis), TensorError) {
  let space = space(tensor)
  let dividend =
    tensor
    |> to_list
    |> list.length

  let initial = FitAcc(divisor: 1, inferring: None)
  let FitAcc(divisor: divisor, inferring: inferring) =
    space
    |> space.elements
    |> list.fold(
      from: initial,
      with: fn(acc, element) {
        let #(axis, size) = element
        case size == -1 {
          False -> FitAcc(..acc, divisor: acc.divisor * size)
          True -> FitAcc(..acc, inferring: Some(axis))
        }
      },
    )

  case dividend % divisor {
    0 ->
      case inferring {
        None -> Ok(tensor)
        Some(infer) -> {
          assert Ok(space) =
            space
            |> space.map_elements(with: fn(element) {
              let #(axis, _) = element
              case axis == infer {
                False -> element
                True -> #(axis, dividend / divisor)
              }
            })
          Ok(Tensor(..tensor, space: space))
        }
      }
    _ -> Error(IncompatibleShape)
  }
}

type FitAcc(axis) {
  FitAcc(divisor: Int, inferring: Option(axis))
}

fn map_data(
  over tensor: Tensor(a, dn, axis),
  with fun: fn(Native) -> Native,
) -> Tensor(a, dn, axis) {
  Tensor(
    ..tensor,
    data: tensor
    |> to_native
    |> fun,
  )
}

if erlang {
  external fn list_to_tuple(List(a)) -> Dynamic =
    "erlang" "list_to_tuple"
}

/// Prints a `Tensor`'s underlying data to standard out.
///
/// Takes a `max_width` (in columns) argument for which the special values `-1`
/// and `0` represent default and no wrapping, respectively.
///
/// If the `meta` argument is `True`, various `Tensor` details will be printed
/// with the data.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/space
/// > type Axis { X Y Z }
/// > assert Ok(space) = space.d2(#(X, 2), #(Y, 2))
/// > try tensor = from_ints(of: [1, 2, 3, 4], into: space)
/// > Ok(print(from: tensor, wrap_at: -1, meta: False))
/// // [[1, 2],
/// //  [3, 4]]
/// Ok(Nil)
///
/// > assert Ok(space) = space.d3(#(X, 2), #(Y, 1), #(Z, 4))
/// > let list = [1, 2, 3, 4, 5, 6, 7, 8]
/// > try tensor = from_ints(of: list, into: space)
/// > Ok(print(from: tensor, wrap_at: 10, meta: True))
/// // Tensor
/// // format: Int64
/// // space: D3 #(X, 2), #(Y, 1), #(Z, 4)
/// // data:
/// // [[[1, 2,
/// //    3, 4]],
/// //  [[5, 6,
/// //    7, 8]]]
/// Ok(Nil)
///
pub fn print(
  from tensor: Tensor(a, dn, axis),
  wrap_at max_width: Int,
  meta meta: Bool,
) -> Nil {
  let string = data_to_string(from: tensor, wrap_at: max_width)
  let string = case rank(tensor) > 0 {
    True -> string
    False ->
      string
      |> string.drop_left(up_to: 1)
      |> string.drop_right(up_to: 1)
  }

  case meta {
    False -> ""
    True -> {
      let format =
        tensor
        |> format
        |> format.to_string
      let space =
        tensor
        |> space
        |> space.to_string
      [
        "Tensor",
        string.append(to: "format: ", suffix: format),
        string.append(to: "space: ", suffix: space),
        "data:",
        "",
      ]
      |> string.join(with: "\n")
    }
  }
  |> string.append(suffix: string)
  |> io.println
}

/// Returns a `String` representation of a `Tensor`'s data.
///
fn data_to_string(
  from tensor: Tensor(a, dn, axis),
  wrap_at max_width: Int,
) -> String {
  let max_width = case max_width < 0 {
    True -> 39
    False -> max_width
  }

  let is_long = case max_width != 0 {
    True -> fn(int) { int > max_width }
    False -> fn(_) { False }
  }

  let #(_, string) =
    tensor
    |> shape
    |> list.drop(up_to: 1)
    |> list.reverse
    |> list.fold(
      from: tensor
      |> to_list
      |> dynamic.from,
      with: fn(acc, size) {
        assert Ok(acc) = dynamic.shallow_list(acc)
        acc
        |> list.sized_chunk(into: size)
        |> dynamic.from
      },
    )
    |> data_to_string_acc(from: DataToStringAcc(depth: 1, is_long: is_long))

  string.concat(["[", string, "]"])
}

type DataToStringAcc {
  DataToStringAcc(depth: Int, is_long: fn(Int) -> Bool)
}

fn data_to_string_acc(
  from acc: DataToStringAcc,
  over list: Dynamic,
) -> #(DataToStringAcc, String) {
  let ws = string.repeat(" ", times: acc.depth)
  let ws_length = string.length(ws)

  assert Ok(string) =
    list
    |> dynamic.any(of: [
      function.compose(
        dynamic.list(of: dynamic.shallow_list),
        result.map(
          over: _,
          with: fn(list) {
            let #(_, strings) =
              list.map_fold(
                over: list,
                from: DataToStringAcc(..acc, depth: acc.depth + 1),
                with: fn(acc, list) {
                  list
                  |> dynamic.from
                  |> data_to_string_acc(from: acc)
                },
              )
            strings
            |> iterator.from_list
            |> iterator.map(with: fn(string) {
              string.concat(["[", string, "]"])
            })
            |> iterator.intersperse(with: string.append(to: ",\n", suffix: ws))
            |> iterator.to_list
            |> string.concat
          },
        ),
      ),
      function.compose(
        dynamic.shallow_list,
        result.map(
          over: _,
          with: fn(list) {
            let #(_, string) =
              list
              |> iterator.from_list
              |> iterator.index
              |> iterator.fold(
                from: #(0, ""),
                with: fn(inner_acc, tuple) {
                  let #(index, item) = tuple
                  let #(line_length, string) = inner_acc
                  let item = item_to_string(item)
                  let item_length = string.length(item) + 1
                  case index == 0 {
                    True -> #(ws_length + item_length, item)
                    False -> {
                      let item_length = item_length
                      let line_length = line_length + item_length
                      case acc.is_long(line_length + ws_length) {
                        True -> #(
                          ws_length + item_length,
                          string.concat([string, ",\n", ws, item]),
                        )
                        False -> #(
                          line_length + 1,
                          string.concat([string, ", ", item]),
                        )
                      }
                    }
                  }
                },
              )
            string
          },
        ),
      ),
    ])

  #(acc, string)
}

fn item_to_string(item: Dynamic) -> String {
  assert Ok(string) =
    item
    |> dynamic.any(of: [
      function.compose(
        dynamic.float,
        result.map(over: _, with: float.to_string),
      ),
      function.compose(dynamic.int, result.map(over: _, with: int.to_string)),
      dynamic.string,
    ])

  string
}

/// Results in the value of the given function on success, or an error on
/// failure.
///
/// Prevents unsafe functions from crashing.
///
fn rescue(
  from fun1: fn() -> a,
  apply fun2: fn(String) -> error,
) -> Result(a, error) {
  do_rescue(fun1, fun2)
}

if erlang {
  fn do_rescue(fun1: fn() -> a, fun2: fn(String) -> error) -> Result(a, error) {
    fun1
    |> erlang.rescue
    |> result.map_error(with: function.compose(decode_crash, fun2))
  }

  fn decode_crash(crash: Crash) -> String {
    case crash {
      erlang.Errored(dynamic) | erlang.Exited(dynamic) | erlang.Thrown(dynamic) ->
        case is_exception(dynamic) {
          True ->
            dynamic
            |> exception_from_dynamic
            |> result.map(with: message)
          False ->
            dynamic
            |> dynamic.string
        }
        |> result.unwrap(or: "")

      _ -> ""
    }
  }

  external type Exception

  fn exception_from_dynamic(from: Dynamic) -> Result(Exception, DecodeErrors) {
    case is_exception(from) {
      True -> Ok(dynamic.unsafe_coerce(from))
      False ->
        Error([
          DecodeError(
            expected: "Exception",
            found: dynamic.classify(from),
            path: [],
          ),
        ])
    }
  }

  external fn message(Exception) -> String =
    "Elixir.Exception" "message"

  external fn is_exception(a) -> Bool =
    "Elixir.Exception" "exception?"
}

/// Replaces an error message string based on a keyword list of errors.
///
/// Crashes if the message isn't found in the list.
///
fn replace_error(
  when message: String,
  found_in list: List(#(error, String)),
) -> error {
  do_replace_error(message, list)
}

if erlang {
  fn do_replace_error(message: String, list: List(#(error, String))) -> error {
    let options = regex.Options(case_insensitive: True, multi_line: False)
    assert Ok(error) =
      list
      |> list.find_map(with: fn(pair) {
        let #(error, test) = pair
        assert Ok(test) = regex.compile(test, with: options)
        case regex.check(with: test, content: message) {
          True -> Ok(error)
          False -> Error(Nil)
        }
      })

    error
  }
}
