import argamak/axis.{Axes, Axis, Infer}
import argamak/format.{Float32, Format, Int32}
import argamak/space.{Space}
import gleam/bool
//import gleam/function
import gleam/io
import gleam/int
import gleam/iterator
import gleam/list
import gleam/map
import gleam/result
import gleam/string
import gleam/string_builder.{StringBuilder}

/// A `Tensor` is a generic container for n-dimensional data structures.
///
pub opaque type Tensor(a) {
  Tensor(data: Native, format: Format(a), space: Space)
}

/// A type for `Native` tensor representations.
///
pub external type Native

/// When a tensor operation cannot succeed.
///
pub type TensorError {
  CannotBroadcast
  IncompatibleAxes
  IncompatibleShape
  InvalidData
  SpaceErrors(space.SpaceErrors)
  ZeroDivision
}

/// A `Result` alias type for tensors.
///
pub type TensorResult(a) =
  Result(Tensor(a), TensorError)

/// A `Result` alias type for `Native` tensor data.
///
pub type NativeResult =
  Result(Native, TensorError)

/// References a space's `Axes` by index.
///
type Indices =
  List(Int)

/// The sizes of a space's `Axes`.
///
type Shape =
  List(Int)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Creation Functions                     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Creates a `Tensor` from a `Float`.
///
/// ## Examples
///
/// ```gleam
/// > from_float(1.) |> print_data
/// 1.
/// Nil
/// ```
///
pub fn from_float(x: Float) -> Tensor(Float32) {
  assert Ok(x) = tensor(from: x, into: space.new(), with: format.float32())
  x
}

/// Creates a `Tensor` from an `Int`.
///
/// ## Examples
///
/// ```gleam
/// > from_int(1) |> print_data
/// 1
/// Nil
/// ```
///
pub fn from_int(x: Int) -> Tensor(Int32) {
  assert Ok(x) = tensor(from: x, into: space.new(), with: format.int32())
  x
}

/// Creates a `Tensor(Int32)` from a `Bool`.
///
/// `True` is represented by `1`, `False` by `0`.
///
/// ## Examples
///
/// ```gleam
/// > from_bool(True) |> print_data
/// 1
/// Nil
///
/// > from_bool(False) |> print_data
/// 0
/// Nil
/// ```
///
pub fn from_bool(x: Bool) -> Tensor(Int32) {
  assert Ok(x) =
    x
    |> bool.to_int
    |> tensor(into: space.new(), with: format.int32())
  x
}

/// Results in a `Tensor` created from a list of floats and placed into a given
/// `Space` on success, or a `TensorError` on failure.
///
/// The `Space` may have a single `Infer` dimension, the size of which will be
/// determined based on the given list. This is useful when working with lists
/// of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1.], into: d1)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [1.],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_floats(of: [1., 2., 3., 4.], into: d2)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2)),
///   [[1., 2.],
///    [3., 4.]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(Infer("X"), Y(2), Z(2))
/// > let xs = [1., 2., 3., 4., 5., 6., 7., 8.]
/// > assert Ok(x) = from_floats(of: xs, into: d3)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2), Z(2)),
///   [[[1., 2.],
///     [3., 4.]],
///    [[5., 6.],
///     [7., 8.]]],
/// )
/// Nil
/// ```
///
pub fn from_floats(
  of xs: List(Float),
  into space: Space,
) -> TensorResult(Float32) {
  tensor(from: xs, into: space, with: format.float32())
}

/// Results in a `Tensor` created from a list of integers and placed into a
/// given `Space` on success, or a `TensorError` on failure.
///
/// The `Space` may have a single `Infer` dimension, the size of which will be
/// determined based on the given list. This is useful when working with lists
/// of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1], into: d1)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(1)),
///   [1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d2)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 2],
///    [3, 4]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(Infer("X"), Y(2), Z(2))
/// > let xs = [1, 2, 3, 4, 5, 6, 7, 8]
/// > assert Ok(x) = from_ints(of: xs, into: d3)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2), Z(2)),
///   [[[1, 2],
///     [3, 4]],
///    [[5, 6],
///     [7, 8]]],
/// )
/// Nil
/// ```
///
pub fn from_ints(of xs: List(Int), into space: Space) -> TensorResult(Int32) {
  tensor(from: xs, into: space, with: format.int32())
}

/// Results in a `Tensor(Int32)` created from a list of booleans and placed into
/// a given `Space` on success, or a `TensorError` on failure.
///
/// `True` is represented by `1`, `False` by `0`.
///
/// The `Space` may have a single `Infer` dimension, the size of which will be
/// determined based on the given list. This is useful when working with lists
/// of unknown length.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_bools(of: [True], into: d1)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(1)),
///   [1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_bools(of: [True, False, True, True], into: d2)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 0],
///    [1, 1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(Infer("X"), Y(2), Z(2))
/// > let xs = [True, False, True, False, False, True, False, True]
/// > assert Ok(x) = from_bools(of: xs, into: d3)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2), Z(2)),
///   [[[1, 0],
///     [1, 0]],
///    [[0, 1],
///     [0, 1]]],
/// )
/// Nil
/// ```
///
pub fn from_bools(of xs: List(Bool), into space: Space) -> TensorResult(Int32) {
  xs
  |> list.map(with: bool.to_int)
  |> tensor(into: space, with: format.int32())
}

/// Results in a `Tensor` created from a `Native` representation on success, or
/// a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y}
/// > import argamak/format
/// > import argamak/space
/// > import gleam/dynamic.{Dynamic}
/// > external fn erlang_tensor(Dynamic) -> Native =
/// >   "Elixir.Nx" "tensor"
/// > let native = erlang_tensor(dynamic.from([[1, 2], [3, 4]]))
/// > assert Ok(d2) = space.d2(X(2), Infer("Y"))
/// > assert Ok(x) = from_native(of: native, into: d2, with: format.int32)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 2],
///    [3, 4]],
/// )
/// Nil
/// ```
///
pub fn from_native(
  of x: Native,
  into space: Space,
  with format: Format(a),
) -> TensorResult(a) {
  x
  |> Tensor(space: space, format: format)
  |> reformat(apply: format)
  |> reshape(into: space)
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reflection Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Returns the `Format` of a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// import argamak/format
/// > format(from_float(0.))
/// format.float32()
///
/// > import argamak/axis.{Infer}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d1)
/// > format(x)
/// format.int32()
/// ```
///
pub fn format(x: Tensor(a)) -> Format(a) {
  x.format
}

/// Returns the `Space` a given `Tensor` is currently in.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > space(from_float(0.)) |> space.axes
/// []
///
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d1)
/// > space(x) |> space.axes
/// [X(3)]
///
/// > assert Ok(d3) = space.d3(X(2), Y(2), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: d3)
/// > space(x) |> space.axes
/// [X(2), Y(2), Z(2)]
/// ```
///
pub fn space(x: Tensor(a)) -> Space {
  x.space
}

/// Returns the `Axes` of a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > axes(from_int(3))
/// []
///
/// > import argamak/axis.{Axis, X}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1., 2., 3.], into: d1)
/// > axes(x)
/// [X(3)]
///
/// > assert Ok(d2) = space.d2(Axis("Alpha", 1), Axis("Omega", 3))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d2)
/// > axes(x)
/// [Axis("Alpha", 1), Axis("Omega", 3)]
/// ```
///
pub fn axes(x: Tensor(a)) -> Axes {
  x
  |> space
  |> space.axes
}

/// Returns the rank of a given `Tensor` as an `Int` representing the number of
/// `Axes`.
///
/// ## Examples
///
/// ```gleam
/// > rank(from_float(0.))
/// 0
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d1)
/// > rank(x)
/// 1
///
/// > assert Ok(d3) = space.d3(X(2), Y(2), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: d3)
/// > rank(x)
/// 3
/// ```
///
pub fn rank(x: Tensor(a)) -> Int {
  x
  |> space
  |> space.degree
}

/// Returns the shape of a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > shape(from_float(0.))
/// []
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d1)
/// > shape(x)
/// [3]
///
/// > assert Ok(d3) = space.d3(X(2), Y(2), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: d3)
/// > shape(x)
/// [2, 2, 2]
/// ```
///
pub fn shape(x: Tensor(a)) -> Shape {
  x
  |> space
  |> space.shape
}

/// Returns the number of values in a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > size(from_float(0.))
/// 1
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d1)
/// > size(x)
/// 3
///
/// > assert Ok(d3) = space.d3(X(2), Y(2), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: d3)
/// > size(x)
/// 8
/// ```
///
pub fn size(x: Tensor(a)) -> Int {
  // Optimized via external implementation.
  x
  |> to_native
  |> do_size
}

if erlang {
  external fn do_size(Native) -> Int =
    "argamak_ffi" "size"
}

if javascript {
  external fn do_size(Native) -> Int =
    "../argamak_ffi.mjs" "size"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Transformation Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Changes the `Format` of a `Tensor`.
///
/// Reformatting from `Float` formats to `Int` formats truncates the data. For
/// consistency, consider using `round`, `floor`, or `ceiling` beforehand.
///
/// Lowering precision may lead to an overflow or underflow, the outcome of
/// which depends on platform and compiler.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/format
/// > reformat(from_int(0), apply: format.float32) |> print
/// Tensor(
///   Format(Float32),
///   Space(),
///   0.,
/// )
/// Nil
///
/// > import argamak/axis.{Infer}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1., 2., 3.], into: d1)
/// > reformat(x, apply: format.int32) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(3)),
///   [1, 2, 3],
/// )
/// Nil
/// ```
///
pub fn reformat(x: Tensor(a), apply format: Format(b)) -> Tensor(b) {
  x
  |> to_native
  |> do_reformat(format.to_native(format))
  |> Tensor(format: format, space: space(x))
}

if erlang {
  external fn do_reformat(Native, format.Native) -> Native =
    "argamak_ffi" "reformat"
}

if javascript {
  external fn do_reformat(Native, format.Native) -> Native =
    "../argamak_ffi.mjs" "reformat"
}

/// Results in a `Tensor` placed into a given `Space` on success, or a
/// `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = reshape(put: from_float(1.), into: d1)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [1.],
/// )
/// Nil
///
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1., 2., 3., 4.], into: d1)
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = reshape(put: x, into: d2)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2)),
///   [[1., 2.],
///    [3., 4.]],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(X(2), Infer("Y"))
/// > let xs = [1., 2., 3., 4., 5., 6., 7., 8.]
/// > assert Ok(x) = from_floats(of: xs, into: d2)
/// > assert Ok(d3) = space.d3(Infer("X"), Y(2), Z(2))
/// > assert Ok(x) = reshape(put: x, into: d3)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2), Z(2)),
///   [[[1., 2.],
///     [3., 4.]],
///    [[5., 6.],
///     [7., 8.]]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Y(1), Z(1))
/// > assert Ok(x) = from_floats(of: [1.], into: d3)
/// > assert Ok(x) = reshape(put: x, into: space.new())
/// > print_data(x)
/// 1.
/// Nil
/// ```
///
pub fn reshape(put x: Tensor(a), into new_space: Space) -> TensorResult(a) {
  try x =
    Tensor(..x, space: new_space)
    |> fit
  let shape = shape(x)
  try native =
    x
    |> to_native
    |> do_reshape(shape)
  Tensor(..x, data: native)
  |> Ok
}

if erlang {
  external fn do_reshape(Native, Shape) -> NativeResult =
    "argamak_ffi" "reshape"
}

if javascript {
  external fn do_reshape(Native, Shape) -> NativeResult =
    "../argamak_ffi.mjs" "reshape"
}

/// Results in a `Tensor` broadcast into a given `Space` on success, or a
/// `TensorError` on failure.
///
/// The new `Space` cannot have fewer `Axes` than the current `Space`.
///
/// Each current `Axis` size must be `1` or equal to its counterpart in the new
/// `Space`. `Axis` compatibility is considered element-wise, tail-first.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(X(3))
/// > assert Ok(x) = broadcast(from: from_int(0), into: d1)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(3)),
///   [0, 0, 0],
/// )
/// Nil
///
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > let x = from_ints(of: [-1], into: d1)
/// > assert Ok(d1) = space.d1(Y(5))
/// > assert Ok(x) = broadcast(from: x, into: d1)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(5)),
///   [-1, -1, -1, -1, -1],
/// )
/// Nil
///
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1., 2., 3.], into: d1)
/// > assert Ok(d2) = space.d2(X(2), Y(3))
/// > assert Ok(x) = broadcast(from: x, into: d2)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(3)),
///   [[1., 2., 3.],
///    [1., 2., 3.]],
/// )
/// Nil
/// ```
///
pub fn broadcast(from x: Tensor(a), into new_space: Space) -> TensorResult(a) {
  try native =
    x
    |> to_native
    |> do_broadcast(space.shape(new_space))
  Tensor(..x, data: native, space: new_space)
  |> Ok
}

if erlang {
  external fn do_broadcast(Native, Shape) -> NativeResult =
    "argamak_ffi" "broadcast"
}

if javascript {
  external fn do_broadcast(Native, Shape) -> NativeResult =
    "../argamak_ffi.mjs" "broadcast"
}

/// A variant of `broadcast` that maps the given `Tensor`'s current `Axes` to
/// arbitrary counterparts in the new `Space`.
///
/// Results in a `Tensor` broadcast into a given `Space` on success, or a
/// `TensorError` on failure.
///
/// The map function allows broadcasting into a `Space` that is incompatible
/// with a standard `broadcast` operation. Any given new `Axis` must not be
/// matched with multiple axes from the `Tensor`'s current `Space`, and the
/// current axes' relative order may be interrupted, but not altered, when axes
/// are translated to their mapped counterparts.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1., 2., 3.], into: d1)
/// > assert Ok(d2) = space.d2(X(3), Y(2))
/// > assert Ok(y) = broadcast_over(from: x, into: d2, with: fn(_) { "X" })
/// > print(y)
/// Tensor(
///   Format(Float32),
///   Space(X(3), Y(2)),
///   [[1., 1.],
///    [2., 2.],
///    [3., 3.]],
/// )
/// Nil
///
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4, 5, 6], into: d2)
/// > assert Ok(d3) = space.d3(X(3), Y(2), Z(2))
/// > assert Ok(y) = broadcast_over(
///     from: x,
///     into: d3,
///     with: fn(a) {
///       case axis.name(a) {
///         "Y" -> "Z"
///         name -> name
///       }
///     },
///   )
/// > print(y)
/// Tensor(
///   Format(Int32),
///   Space(X(3), Y(2), Z(2))
///   [[[1, 2],
///     [1, 2]],
///    [[3, 4],
///     [3, 4]],
///    [[5, 6],
///     [5, 6]]],
/// )
/// Nil
///
/// > assert Ok(y) = broadcast_over(
///     from: x,
///     into: d3,
///     with: axis.name,
///   )
/// > print(y)
/// Tensor(
///   Format(Int32),
///   Space(X(3), Y(2), Z(2)),
///   [[[1, 1],
///     [2, 2]],
///    [[3, 3],
///     [4, 4]],
///    [[5, 5],
///     [6, 6]]],
/// )
/// Nil
/// ```
///
pub fn broadcast_over(
  from x: Tensor(a),
  into new_space: Space,
  with space_map: fn(Axis) -> String,
) -> TensorResult(a) {
  let new_axes = space.axes(new_space)

  // try mapped_axes =
  //   x
  //   |> axes
  //   |> list.map(
  //     with: space_map
  //     |> function.compose(fn(name) {
  //       new_axes
  //       |> list.find_map(with: fn(axis) {
  //         case axis.name(axis) == name {
  //           True ->
  //             #(name, axis.size(axis))
  //             |> Ok
  //           False -> Error(Nil)
  //         }
  //       })
  //       |> result.replace_error(IncompatibleAxes)
  //     }),
  //   )
  //   |> result.all
  try mapped_axes =
    result.all({
      use axis <- list.map(axes(x))
      let name = space_map(axis)
      {
        use axis <- list.find_map(new_axes)
        case axis.name(axis) == name {
          True ->
            #(name, axis.size(axis))
            |> Ok
          False -> Error(Nil)
        }
      }
      |> result.replace_error(IncompatibleAxes)
    })
  let axis_map = map.from_list(mapped_axes)

  // TODO: use higher level functions?
  // let pre_shape =
  //   new_axes
  //   |> list.map(with: fn(axis) {
  //     axis_map
  //     |> map.get(axis.name(axis))
  //     |> result.unwrap(or: 1)
  //   })
  let pre_shape = {
    use axis <- list.map(new_axes)
    axis_map
    |> map.get(axis.name(axis))
    |> result.unwrap(or: 1)
  }

  let shape = space.shape(new_space)
  try native =
    x
    |> to_native
    |> do_reshape(pre_shape)
  try native = do_broadcast(native, shape)
  Tensor(..x, data: native, space: new_space)
  |> Ok
}

/// Removes from the given `Tensor` axes of size `1` for which `filter` returns
/// `True`.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > squeeze(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [3], into: d1)
/// > squeeze(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > squeeze(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1)),
///   [3],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [1, 2], into: d3)
/// > squeeze(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 2],
/// )
/// Nil
///
/// > squeeze(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[1, 2]],
/// )
/// Nil
/// ```
///
pub fn squeeze(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  // let filter = fn(axis) {
  //   case axis.size(axis) {
  //     1 -> filter(axis)
  //     _else -> False
  //   }
  // }
  // reducible_over_axes(do_squeeze, x, filter, Away)
  use axis <- reducible_over_axes(do_squeeze, x, _, Away)
  case axis.size(axis) {
    1 -> filter(axis)
    _else -> False
  }
}

if erlang {
  external fn do_squeeze(Native, Indices) -> Native =
    "argamak_ffi" "squeeze"
}

if javascript {
  external fn do_squeeze(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "squeeze"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Logical Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Equality is represented by `1`, inequality by `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = equal(is: a, to: from_int(4))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = equal(is: a, to: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[0, 1],
///    [1, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > equal(is: b, to: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > equal(is: b, to: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn equal(is a: Tensor(a), to b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_equal, a, b)
}

if erlang {
  external fn do_equal(Native, Native) -> NativeResult =
    "argamak_ffi" "equal"
}

if javascript {
  external fn do_equal(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "equal"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Inequality is represented by `1`, equality by `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = not_equal(is: a, to: from_int(4))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = not_equal(is: a, to: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 0],
///    [0, 1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > not_equal(is: b, to: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > not_equal(is: b, to: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn not_equal(is a: Tensor(a), to b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_not_equal, a, b)
}

if erlang {
  external fn do_not_equal(Native, Native) -> NativeResult =
    "argamak_ffi" "not_equal"
}

if javascript {
  external fn do_not_equal(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "not_equal"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values in the first `Tensor` that are greater than those in the second are
/// represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = greater(is: a, than: from_int(4))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = greater(is: a, than: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 0],
///    [0, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > greater(is: b, than: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > greater(is: b, than: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn greater(is a: Tensor(a), than b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_greater, a, b)
}

if erlang {
  external fn do_greater(Native, Native) -> NativeResult =
    "argamak_ffi" "greater"
}

if javascript {
  external fn do_greater(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "greater"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values in the first `Tensor` that are greater than or equal to those in the
/// second are represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = greater_or_equal(is: a, to: from_int(4))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = greater_or_equal(is: a, to: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 1],
///    [1, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > greater_or_equal(is: b, to: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > greater_or_equal(is: b, to: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn greater_or_equal(is a: Tensor(a), to b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_greater_or_equal, a, b)
}

if erlang {
  external fn do_greater_or_equal(Native, Native) -> NativeResult =
    "argamak_ffi" "greater_or_equal"
}

if javascript {
  external fn do_greater_or_equal(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "greater_or_equal"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values in the first `Tensor` that are less than those in the second are
/// represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = less(is: a, than: from_int(5))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = less(is: a, than: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[0, 0],
///    [0, 1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > less(is: b, than: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > less(is: b, than: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn less(is a: Tensor(a), than b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_less, a, b)
}

if erlang {
  external fn do_less(Native, Native) -> NativeResult =
    "argamak_ffi" "less"
}

if javascript {
  external fn do_less(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "less"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values in the first `Tensor` that are less than or equal to those in the
/// second are represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [5, 4], into: d1)
/// > assert Ok(x) = less_or_equal(is: a, to: from_int(5))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [4, 4, 5, 5], into: d2)
/// > assert Ok(x) = less_or_equal(is: a, to: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[0, 1],
///    [1, 1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 6], into: d3)
/// > less_or_equal(is: b, to: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > less_or_equal(is: b, to: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn less_or_equal(is a: Tensor(a), to b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_less_or_equal, a, b)
}

if erlang {
  external fn do_less_or_equal(Native, Native) -> NativeResult =
    "argamak_ffi" "less_or_equal"
}

if javascript {
  external fn do_less_or_equal(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "less_or_equal"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values that are nonzero in both the first and second `Tensor`s are
/// represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [9, 0], into: d1)
/// > assert Ok(x) = logical_and(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 0], into: d2)
/// > assert Ok(x) = logical_and(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[0, 0],
///    [1, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > logical_and(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > logical_and(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn logical_and(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_logical_and, a, b)
}

if erlang {
  external fn do_logical_and(Native, Native) -> NativeResult =
    "argamak_ffi" "logical_and"
}

if javascript {
  external fn do_logical_and(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "logical_and"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values that are nonzero in either the first or second `Tensor`, or both, are
/// represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [9, 0], into: d1)
/// > assert Ok(x) = logical_or(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 0], into: d2)
/// > assert Ok(x) = logical_or(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 1],
///    [1, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > logical_or(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > logical_or(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn logical_or(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_logical_or, a, b)
}

if erlang {
  external fn do_logical_or(Native, Native) -> NativeResult =
    "argamak_ffi" "logical_or"
}

if javascript {
  external fn do_logical_or(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "logical_or"
}

/// Results in a `Tensor` that is the element-wise comparison of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// Values that are nonzero in either the first or second `Tensor`, but not
/// both, are represented by `1`, otherwise `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [9, 0], into: d1)
/// > assert Ok(x) = logical_xor(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 1],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 0], into: d2)
/// > assert Ok(x) = logical_xor(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 1],
///    [0, 0]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > logical_xor(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > logical_xor(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn logical_xor(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_logical_xor, a, b)
}

if erlang {
  external fn do_logical_xor(Native, Native) -> NativeResult =
    "argamak_ffi" "logical_xor"
}

if javascript {
  external fn do_logical_xor(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "logical_xor"
}

/// Returns the element-wise logical opposite of the given `Tensor`.
///
/// Values that are nonzero are represented by `1`, otherwise `0`, with `Format`
/// retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > logical_not(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.3], into: d1)
/// > logical_not(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [-1, 8, 0], into: d3)
/// > logical_not(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(3), Z(1)),
///   [[[0],
///     [0],
///     [1]]],
/// )
/// Nil
/// ```
///
pub fn logical_not(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_logical_not(to_native(x)))
}

if erlang {
  external fn do_logical_not(Native) -> Native =
    "argamak_ffi" "logical_not"
}

if javascript {
  external fn do_logical_not(Native) -> Native =
    "../argamak_ffi.mjs" "logical_not"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Arithmetic Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a `Tensor` that is the element-wise addition of the given tensors
/// on success (broadcast as needed), or a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [0, 9], into: d1)
/// > assert Ok(x) = add(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [ 3, 12],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 0], into: d2)
/// > assert Ok(x) = add(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 0, 13],
///    [ 5,  9]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > add(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > add(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn add(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_add, a, b)
}

if erlang {
  external fn do_add(Native, Native) -> NativeResult =
    "argamak_ffi" "add"
}

if javascript {
  external fn do_add(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "add"
}

/// Results in a `Tensor` that is the element-wise subtraction of one `Tensor`
/// from another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [0, 9], into: d1)
/// > assert Ok(x) = subtract(from: a, value: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [-3,  6],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 0], into: d2)
/// > assert Ok(x) = subtract(from: a, value: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 0,  5],
///    [-5,  9]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > subtract(from: b, value: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > subtract(from: b, value: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn subtract(from a: Tensor(a), value b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_subtract, a, b)
}

if erlang {
  external fn do_subtract(Native, Native) -> NativeResult =
    "argamak_ffi" "subtract"
}

if javascript {
  external fn do_subtract(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "subtract"
}

/// Results in a `Tensor` that is the element-wise multiplication of the given
/// tensors on success (broadcast as needed), or a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = multiply(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [ 3, 27],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 9], into: d2)
/// > assert Ok(x) = multiply(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 0, 36],
///    [ 5, 81]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > multiply(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > multiply(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn multiply(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_multiply, a, b)
}

if erlang {
  external fn do_multiply(Native, Native) -> NativeResult =
    "argamak_ffi" "multiply"
}

if javascript {
  external fn do_multiply(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "multiply"
}

/// Results in a `Tensor` that is the element-wise division of one `Tensor` by
/// another on success (broadcast as needed), or a `TensorError` on failure.
///
/// As with Gleam's operators, division by zero returns zero.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = divide(from: a, by: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 3],
/// )
/// Nil
///
/// > let a = reformat(a, apply: format.float32())
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_floats(of: [0., 4., 5., 9.], into: d2)
/// > assert Ok(x) = divide(from: a, by: b)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2)),
///   [[  0., 2.25],
///    [ 0.2,   1.]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_floats(of: [4., 5., 0.], into: d3)
/// > divide(from: b, by: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > divide(from: b, by: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn divide(from a: Tensor(a), by b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_divide, a, _)
  |> permit_zero(in: b)
}

if erlang {
  external fn do_divide(Native, Native) -> NativeResult =
    "argamak_ffi" "divide"
}

if javascript {
  external fn do_divide(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "divide"
}

/// A variant of `divide` that results in a `TensorError` if any value of the
/// divisor is zero.
///
/// Results in a `Tensor` that is the element-wise division of one `Tensor` by
/// another on success (broadcast as needed), or a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = try_divide(from: a, by: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 3],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 9], into: d2)
/// > try_divide(from: a, by: b)
/// Error(ZeroDivision)
/// ```
///
pub fn try_divide(from a: Tensor(a), by b: Tensor(a)) -> TensorResult(a) {
  try b = all_nonzero(b)
  divide(from: a, by: b)
}

/// Results in a `Tensor` that is the element-wise remainder when dividing one
/// `Tensor` by another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// As with Gleam's operators, division by zero returns zero.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [13, -13], into: d1)
/// > assert Ok(x) = remainder(from: a, divided_by: from_int(0))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [3, 3, -3, -3], into: d2)
/// > assert Ok(x) = remainder(from: a, divided_by: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 1, -1],
///    [ 1, -1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > remainder(from: b, divided_by: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > remainder(from: b, divided_by: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn remainder(from a: Tensor(a), divided_by b: Tensor(a)) -> TensorResult(a) {
  do_remainder(a, b)
}

if erlang {
  fn do_remainder(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
    broadcastable(erlang_remainder, a, _)
    |> permit_zero(in: b)
  }

  external fn erlang_remainder(Native, Native) -> NativeResult =
    "argamak_ffi" "remainder"
}

if javascript {
  fn do_remainder(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
    try adjust = sign_not_equal(a, b)
    try adjust = multiply(adjust, b)
    try x = modulo(from: a, divided_by: b)
    subtract(from: x, value: adjust)
  }
}

/// A variant of `remainder` that results in a `TensorError` if any value of the
/// divisor is zero.
///
/// Results in a `Tensor` that is the element-wise remainder when dividing one
/// `Tensor` by another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = try_remainder(from: a, divided_by: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 9], into: d2)
/// > try_remainder(from: a, divided_by: b)
/// Error(ZeroDivision)
/// ```
///
pub fn try_remainder(
  from a: Tensor(a),
  divided_by b: Tensor(a),
) -> TensorResult(a) {
  try b = all_nonzero(b)
  remainder(from: a, divided_by: b)
}

/// Results in a `Tensor` that is the element-wise modulus when dividing one
/// `Tensor` by another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// As with Gleam's operators, division by zero returns zero.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [13, -13], into: d1)
/// > assert Ok(x) = modulo(from: a, divided_by: from_int(0))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [0, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [3, 3, -3, -3], into: d2)
/// > assert Ok(x) = modulo(from: a, divided_by: b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 1, 2],
///    [ -2, -1]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > modulo(from: b, divided_by: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > modulo(from: b, divided_by: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn modulo(from a: Tensor(a), divided_by b: Tensor(a)) -> TensorResult(a) {
  do_modulo(a, b)
}

if erlang {
  fn do_modulo(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
    try adjust = sign_not_equal(a, b)
    try adjust = multiply(adjust, b)
    try x = remainder(from: a, divided_by: b)
    add(x, adjust)
  }
}

if javascript {
  fn do_modulo(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
    broadcastable(javascript_modulo, a, _)
    |> permit_zero(in: b)
  }

  external fn javascript_modulo(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "modulo"
}

/// A variant of `modulo` that results in a `TensorError` if any value of the
/// divisor is zero.
///
/// Results in a `Tensor` that is the element-wise modulus when dividing one
/// `Tensor` by another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = try_modulo(from: a, divided_by: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, 9], into: d2)
/// > try_modulo(from: a, divided_by: b)
/// Error(ZeroDivision)
/// ```
///
pub fn try_modulo(from a: Tensor(a), divided_by b: Tensor(a)) -> TensorResult(a) {
  try b = all_nonzero(b)
  modulo(from: a, divided_by: b)
}

/// Results in a `Tensor` that is the element-wise raising of one `Tensor` to
/// the power of another on success (broadcast as needed), or a `TensorError` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = power(raise: a, to_the: from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [  1, 729],
/// )
/// Nil
///
/// > let a = reformat(a, apply: format.float32())
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_floats(of: [0., 0.4, 0.5, 0.9], into: d2)
/// > assert Ok(x) = power(raise: a, to_the: b)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(2), Y(2)),
///   [[   1., 2.408],
///    [   1., 7.225]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_floats(of: [4., 5., 0.], into: d3)
/// > power(raise: b, to_the: c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > power(raise: b, to_the: c)
/// Error(CannotBroadcast)
/// ```
///
pub fn power(raise a: Tensor(a), to_the b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_power, a, b)
}

if erlang {
  external fn do_power(Native, Native) -> NativeResult =
    "argamak_ffi" "power"
}

if javascript {
  external fn do_power(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "power"
}

/// Results in a `Tensor` that is the element-wise maximum of the given tensors
/// on success (broadcast as needed), or a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = max(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [3, 9],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, -9], into: d2)
/// > assert Ok(x) = max(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 9],
///    [5, 9]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > max(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > max(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn max(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_max, a, b)
}

if erlang {
  external fn do_max(Native, Native) -> NativeResult =
    "argamak_ffi" "max"
}

if javascript {
  external fn do_max(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "max"
}

/// Results in a `Tensor` that is the element-wise minimum of the given tensors
/// on success (broadcast as needed), or a `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("Y"))
/// > assert Ok(a) = from_ints(of: [1, 9], into: d1)
/// > assert Ok(x) = min(a, from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(Y(2)),
///   [1, 3],
/// )
/// Nil
///
/// > assert Ok(d2) = space.d2(Infer("X"), Y(2))
/// > assert Ok(b) = from_ints(of: [0, 4, 5, -9], into: d2)
/// > assert Ok(x) = min(a, b)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[ 0,  4],
///    [ 1, -9]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(c) = from_ints(of: [4, 5, 0], into: d3)
/// > min(b, c)
/// Error(SpaceErrors([
///   SpaceError(CannotMerge, X(2)),
///   SpaceError(CannotMerge, Y(2)),
/// ]))
///
/// > assert Ok(d3) = space.d3(Z(1), Infer("X"), Y(1))
/// > assert Ok(c) = reshape(put: c, into: d3)
/// > min(b, c)
/// Error(CannotBroadcast)
/// ```
///
pub fn min(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  broadcastable(do_min, a, b)
}

if erlang {
  external fn do_min(Native, Native) -> NativeResult =
    "argamak_ffi" "min"
}

if javascript {
  external fn do_min(Native, Native) -> NativeResult =
    "../argamak_ffi.mjs" "min"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Basic Math Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Returns the element-wise absolute value of the given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > absolute_value(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.3], into: d1)
/// > absolute_value(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.3],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [-1, 8, 0], into: d3)
/// > absolute_value(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(3), Z(1)),
///   [[[1],
///     [8],
///     [0]]],
/// )
/// Nil
/// ```
///
pub fn absolute_value(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_absolute_value(to_native(x)))
}

if erlang {
  external fn do_absolute_value(Native) -> Native =
    "argamak_ffi" "absolute_value"
}

if javascript {
  external fn do_absolute_value(Native) -> Native =
    "../argamak_ffi.mjs" "absolute_value"
}

/// Returns the element-wise negation of the given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > negate(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   -3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.3], into: d1)
/// > negate(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.3],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [-1, 8, 0], into: d3)
/// > negate(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(3), Z(1)),
///   [[[ 1],
///     [-8],
///     [ 0]]],
/// )
/// Nil
/// ```
///
pub fn negate(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_negate(to_native(x)))
}

if erlang {
  external fn do_negate(Native) -> Native =
    "argamak_ffi" "negate"
}

if javascript {
  external fn do_negate(Native) -> Native =
    "../argamak_ffi.mjs" "negate"
}

/// Returns an element-wise indication of the sign of the given `Tensor`.
///
/// Positive numbers are represented by `1`, negative numbers by `-1`, and zero
/// by `0`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > sign(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.3], into: d1)
/// > sign(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [-1.],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [-1, 8, 0], into: d3)
/// > sign(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(3), Z(1)),
///   [[[-1],
///     [ 1],
///     [ 0]]],
/// )
/// Nil
/// ```
///
pub fn sign(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_sign(to_native(x)))
}

if erlang {
  external fn do_sign(Native) -> Native =
    "argamak_ffi" "sign"
}

if javascript {
  external fn do_sign(Native) -> Native =
    "../argamak_ffi.mjs" "sign"
}

/// Returns the element-wise ceiling of the given `Tensor`, with `Format`
/// retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > ceiling(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.5], into: d1)
/// > ceiling(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [-1.2, 7.8, 0.], into: d3)
/// > ceiling(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[-1.],
///     [ 8.],
///     [ 0.]]],
/// )
/// Nil
/// ```
///
pub fn ceiling(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_ceiling(to_native(x)))
}

if erlang {
  external fn do_ceiling(Native) -> Native =
    "argamak_ffi" "ceiling"
}

if javascript {
  external fn do_ceiling(Native) -> Native =
    "../argamak_ffi.mjs" "ceiling"
}

/// Returns the element-wise floor of the given `Tensor`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > floor(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.5], into: d1)
/// > floor(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [-1.],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [-1.2, 7.8, 0.], into: d3)
/// > floor(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[-2.],
///     [ 7.],
///     [ 0.]]],
/// )
/// Nil
/// ```
///
pub fn floor(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_floor(to_native(x)))
}

if erlang {
  external fn do_floor(Native) -> Native =
    "argamak_ffi" "floor"
}

if javascript {
  external fn do_floor(Native) -> Native =
    "../argamak_ffi.mjs" "floor"
}

/// Returns the element-wise rounding of the given `Tensor`, with `Format`
/// retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > round(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   3,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.5], into: d1)
/// > round(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [-1.],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [-1.2, 7.8, 0.], into: d3)
/// > round(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[-1.],
///     [ 8.],
///     [ 0.]]],
/// )
/// Nil
/// ```
///
pub fn round(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_round(to_native(x)))
}

if erlang {
  external fn do_round(Native) -> Native =
    "argamak_ffi" "round"
}

if javascript {
  external fn do_round(Native) -> Native =
    "../argamak_ffi.mjs" "round"
}

/// Returns the element-wise natural exponential (Euler's number raised to the
/// power of `x`) of the given `Tensor`, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > let x = from_int(3)
/// > exp(x) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   20,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [-0.5], into: d1)
/// > exp(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.223],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [-1.2, 7.8, 0.], into: d3)
/// > exp(x) |> print
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[   0.301],
///     [2440.603],
///     [      1.]]],
/// )
/// Nil
/// ```
///
pub fn exp(x: Tensor(a)) -> Tensor(a) {
  Tensor(..x, data: do_exp(to_native(x)))
}

if erlang {
  external fn do_exp(Native) -> Native =
    "argamak_ffi" "exp"
}

if javascript {
  external fn do_exp(Native) -> Native =
    "../argamak_ffi.mjs" "exp"
}

/// Results in the element-wise square root of the given `Tensor` on success, or
/// a `TensorError` on failure, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(x) = square_root(from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1.5], into: d1)
/// > assert Ok(x) = square_root(x)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [1.225],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [1.2, 7.8, 0.], into: d3)
/// > assert Ok(x) = square_root(x)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[1.095],
///     [2.793],
///     [   0.]]],
/// )
/// Nil
///
/// > square_root(from_float(-0.1))
/// Error(InvalidData)
/// ```
///
pub fn square_root(x: Tensor(a)) -> TensorResult(a) {
  try native =
    x
    |> to_native
    |> do_square_root
  Tensor(..x, data: native)
  |> Ok
}

if erlang {
  external fn do_square_root(Native) -> NativeResult =
    "argamak_ffi" "square_root"
}

if javascript {
  external fn do_square_root(Native) -> NativeResult =
    "../argamak_ffi.mjs" "square_root"
}

/// Results in the element-wise natural logarithm of the given `Tensor` on
/// success, or a `TensorError` on failure, with `Format` retained.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(x) = ln(from_int(3))
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_floats(of: [1.5], into: d1)
/// > assert Ok(x) = ln(x)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1)),
///   [0.405],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_floats(of: [1.2, 7.8, 0.], into: d3)
/// > assert Ok(x) = ln(x)
/// > print(x)
/// Tensor(
///   Format(Float32),
///   Space(X(1), Y(3), Z(1)),
///   [[[    0.182],
///     [    2.054],
///     [-Infinity]]],
/// )
/// Nil
///
/// > ln(from_float(-0.1))
/// Error(InvalidData)
/// ```
///
pub fn ln(x: Tensor(a)) -> TensorResult(a) {
  try native =
    x
    |> to_native
    |> do_ln
  Tensor(..x, data: native)
  |> Ok
}

if erlang {
  external fn do_ln(Native) -> NativeResult =
    "argamak_ffi" "ln"
}

if javascript {
  external fn do_ln(Native) -> NativeResult =
    "../argamak_ffi.mjs" "ln"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reduction Functions                    //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Reduces the given `Tensor` over select axes to `1` if all values across
/// those axes are nonzero, otherwise `0`, with `Format` retained.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 0], into: d1)
/// > all(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > all(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [1, 2], into: d3)
/// > all(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > all(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[1, 1]],
/// )
/// Nil
/// ```
///
pub fn all(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_all, x, filter, Away)
}

if erlang {
  external fn do_all(Native, Indices) -> Native =
    "argamak_ffi" "all"
}

if javascript {
  external fn do_all(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "all"
}

/// A variant of `all` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [0, 2], into: d3)
/// > in_situ_all(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[0],
///     [1]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_all(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_all, x, filter, InSitu)
}

/// Reduces the given `Tensor` over select axes to `1` if any values across
/// those axes are nonzero, otherwise `0`, with `Format` retained.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 0], into: d1)
/// > any(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > any(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [1, 0],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [0, 0], into: d3)
/// > any(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > any(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[0, 0]],
/// )
/// Nil
/// ```
///
pub fn any(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_any, x, filter, Away)
}

if erlang {
  external fn do_any(Native, Indices) -> Native =
    "argamak_ffi" "any"
}

if javascript {
  external fn do_any(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "any"
}

/// A variant of `any` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(1))
/// > assert Ok(x) = from_ints(of: [0, 2], into: d3)
/// > in_situ_any(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[0],
///     [1]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_any(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_any, x, filter, InSitu)
}

/// Reduces the given `Tensor` over a select `Axis` to the lowest index of the
/// max value across that `Axis`, with `Format` retained.
///
/// The first `Axis` for which the given `find` function returns `True` is
/// selected for reduction and will be removed from the reduced tensor's
/// `Space`.
///
/// If the `find` function returns `False` for every `Axis`, the `Tensor` will
/// be flattened and the operation applied over the remaining `Axis`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 0], into: d1)
/// > arg_max(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > arg_max(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d3)
/// > arg_max(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(Y(2), Z(2)),
///   [[0, 0],
///    [0, 0]],
/// )
/// Nil
///
/// > arg_max(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[1, 0]],
/// )
/// Nil
/// ```
///
pub fn arg_max(from x: Tensor(a), with find: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axis(do_arg_max, x, find, Away)
}

if erlang {
  external fn do_arg_max(Native, Int) -> Native =
    "argamak_ffi" "arg_max"
}

if javascript {
  external fn do_arg_max(Native, Int) -> Native =
    "../argamak_ffi.mjs" "arg_max"
}

/// A variant of `arg_max` that preserves all axes from the given `Tensor`.
///
/// An `Axis` for which the given `find` function returns `True` will retain a
/// size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d3)
/// > in_situ_arg_max(x, with: fn(a) { axis.name(a) == "Y" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(2)),
///   [[[1, 0]]],
/// )
/// Nil
///
/// > in_situ_arg_max(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[1],
///     [0]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_arg_max(
  from x: Tensor(a),
  with find: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axis(do_arg_max, x, find, InSitu)
}

/// Reduces the given `Tensor` over a select `Axis` to the lowest index of the
/// min value across that `Axis`, with `Format` retained.
///
/// The first `Axis` for which the given `find` function returns `True` is
/// selected for reduction and will be removed from the reduced tensor's
/// `Space`.
///
/// If the `find` function returns `False` for every `Axis`, the `Tensor` will
/// be flattened and the operation applied over the remaining `Axis`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 0], into: d1)
/// > arg_min(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > arg_min(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d3)
/// > arg_min(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(Y(2), Z(2)),
///   [[0, 0],
///    [0, 0]],
/// )
/// Nil
///
/// > arg_min(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[0, 1]],
/// )
/// Nil
/// ```
///
pub fn arg_min(from x: Tensor(a), with find: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axis(do_arg_min, x, find, Away)
}

if erlang {
  external fn do_arg_min(Native, Int) -> Native =
    "argamak_ffi" "arg_min"
}

if javascript {
  external fn do_arg_min(Native, Int) -> Native =
    "../argamak_ffi.mjs" "arg_min"
}

/// A variant of `arg_min` that preserves all axes from the given `Tensor`.
///
/// An `Axis` for which the given `find` function returns `True` will retain a
/// size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d3)
/// > in_situ_arg_min(x, with: fn(a) { axis.name(a) == "Y" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(2)),
///   [[[0, 1]]],
/// )
/// Nil
///
/// > in_situ_arg_min(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[0],
///     [1]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_arg_min(
  from x: Tensor(a),
  with find: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axis(do_arg_min, x, find, InSitu)
}

/// Reduces the given `Tensor` over select axes to the max value across those
/// axes.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 2], into: d1)
/// > max_over(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   2,
/// )
/// Nil
///
/// > max_over(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [-1,  2],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > max_over(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   4,
/// )
/// Nil
///
/// > max_over(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[4, 3]],
/// )
/// Nil
/// ```
///
pub fn max_over(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_max_over, x, filter, Away)
}

if erlang {
  external fn do_max_over(Native, Indices) -> Native =
    "argamak_ffi" "max_over"
}

if javascript {
  external fn do_max_over(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "max_over"
}

/// A variant of `max_over` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > in_situ_max_over(x, with: fn(a) { axis.name(a) == "Y" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(2)),
///   [[[3, 4]]],
/// )
/// Nil
///
/// > in_situ_max_over(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[4],
///     [3]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_max_over(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_max_over, x, filter, InSitu)
}

/// Reduces the given `Tensor` over select axes to the min value across those
/// axes.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 2], into: d1)
/// > min_over(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   -1,
/// )
/// Nil
///
/// > min_over(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [-1,  2],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > min_over(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > min_over(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[1, 2]],
/// )
/// Nil
/// ```
///
pub fn min_over(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_min_over, x, filter, Away)
}

if erlang {
  external fn do_min_over(Native, Indices) -> Native =
    "argamak_ffi" "min_over"
}

if javascript {
  external fn do_min_over(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "min_over"
}

/// A variant of `min_over` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > in_situ_min_over(x, with: fn(a) { axis.name(a) == "Y" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(2)),
///   [[[1, 2]]],
/// )
/// Nil
///
/// > in_situ_min_over(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[1],
///     [2]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_min_over(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_min_over, x, filter, InSitu)
}

/// Reduces the given `Tensor` over select axes to the sum of the values across
/// those axes.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 2], into: d1)
/// > sum(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   1,
/// )
/// Nil
///
/// > sum(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [-1,  2],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [2, 4, 3, 0], into: d3)
/// > sum(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   9,
/// )
/// Nil
///
/// > sum(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[6, 3]],
/// )
/// Nil
/// ```
///
pub fn sum(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_sum, x, filter, Away)
}

if erlang {
  external fn do_sum(Native, Indices) -> Native =
    "argamak_ffi" "sum"
}

if javascript {
  external fn do_sum(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "sum"
}

/// A variant of `sum` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [2, 4, 3, 0], into: d3)
/// > in_situ_sum(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(1)),
///   [[[9]]],
/// )
/// Nil
///
/// > in_situ_sum(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[6],
///     [3]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_sum(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_sum, x, filter, InSitu)
}

/// Reduces the given `Tensor` over select axes to the product of the values
/// across those axes.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 2], into: d1)
/// > product(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   -2,
/// )
/// Nil
///
/// > product(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [-1,  2],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > product(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   24,
/// )
/// Nil
///
/// > product(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[4, 6]],
/// )
/// Nil
/// ```
///
pub fn product(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_product, x, filter, Away)
}

if erlang {
  external fn do_product(Native, Indices) -> Native =
    "argamak_ffi" "product"
}

if javascript {
  external fn do_product(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "product"
}

/// A variant of `product` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_ints(of: [1, 4, 3, 2], into: d3)
/// > in_situ_product(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(1)),
///   [[[24]]],
/// )
/// Nil
///
/// > in_situ_product(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[4],
///     [6]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_product(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_product, x, filter, InSitu)
}

/// Reduces the given `Tensor` over select axes to the mean of the values across
/// those axes.
///
/// Any `Axis` for which the given `filter` function returns `True` is selected
/// for reduction and will be removed from the reduced tensor's `Space`.
///
/// If the `filter` function returns `False` for every `Axis`, all axes will be
/// retained and the operation applied to every value of the `Tensor`
/// individually.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [-1, 2], into: d1)
/// > mean(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   0,
/// )
/// Nil
///
/// > mean(x, with: fn(_) { False }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(2)),
///   [-1,  2],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_floats(of: [1., 4., 3., 2.], into: d3)
/// > mean(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(),
///   2.5,
/// )
/// Nil
///
/// > mean(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2)),
///   [[2.5, 2.5]],
/// )
/// Nil
/// ```
///
pub fn mean(from x: Tensor(a), with filter: fn(Axis) -> Bool) -> Tensor(a) {
  reducible_over_axes(do_mean, x, filter, Away)
}

if erlang {
  external fn do_mean(Native, Indices) -> Native =
    "argamak_ffi" "mean"
}

if javascript {
  external fn do_mean(Native, Indices) -> Native =
    "../argamak_ffi.mjs" "mean"
}

/// A variant of `mean` that preserves all axes from the given `Tensor`.
///
/// Any `Axis` for which the given `filter` function returns `True` will retain
/// a size of `1` after the operation.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > import argamak/space
/// > assert Ok(d3) = space.d3(X(1), Infer("Y"), Z(2))
/// > assert Ok(x) = from_floats(of: [1., 4., 3., 2.], into: d3)
/// > in_situ_product(x, with: fn(_) { True }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(1), Z(1)),
///   [[[2.5]]],
/// )
/// Nil
///
/// > in_situ_product(x, with: fn(a) { axis.name(a) == "Z" }) |> print
/// Tensor(
///   Format(Int32),
///   Space(X(1), Y(2), Z(1)),
///   [[[2.5],
///     [2.5]]],
/// )
/// Nil
/// ```
///
pub fn in_situ_mean(
  from x: Tensor(a),
  with filter: fn(Axis) -> Bool,
) -> Tensor(a) {
  reducible_over_axes(do_mean, x, filter, InSitu)
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Slicing & Joining Functions            //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// TODO
// concat
// take
//   take_along_axis
//   gather
// reverse
// slice
// put_slice
// split
// tile
// stack
// unstack

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Conversion Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a `Float` converted from a shapeless `Tensor` on success, or a
/// `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > to_float(from_float(0.))
/// Ok(0.)
///
/// > to_float(from_int(0))
/// Ok(0.)
///
/// import argamak/axis.{Infer}
/// import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1], into: d1)
/// > to_float(x)
/// Error(IncompatibleShape)
/// ```
///
pub fn to_float(x: Tensor(a)) -> Result(Float, TensorError) {
  x
  |> to_native
  |> do_to_float
}

if erlang {
  external fn do_to_float(Native) -> Result(Float, TensorError) =
    "argamak_ffi" "to_float"
}

if javascript {
  external fn do_to_float(Native) -> Result(Float, TensorError) =
    "../argamak_ffi.mjs" "to_float"
}

/// Results in a `Int` converted from a shapeless `Tensor` on success, or a
/// `TensorError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > to_int(from_float(0.))
/// Ok(0)
///
/// > to_int(from_int(0))
/// Ok(0)
///
/// import argamak/axis.{Infer}
/// import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1], into: d1)
/// > to_int(x)
/// Error(IncompatibleShape)
/// ```
///
pub fn to_int(x: Tensor(a)) -> Result(Int, TensorError) {
  x
  |> to_native
  |> do_to_int
}

if erlang {
  external fn do_to_int(Native) -> Result(Int, TensorError) =
    "argamak_ffi" "to_int"
}

if javascript {
  external fn do_to_int(Native) -> Result(Int, TensorError) =
    "../argamak_ffi.mjs" "to_int"
}

/// Results in a `Bool` converted from a shapeless `Tensor` on success, or a
/// `TensorError` on failure.
///
/// Nonzero values become `True`, otherwise `False`.
///
/// ## Examples
///
/// ```gleam
/// > to_bool(from_float(0.))
/// Ok(False)
///
/// > to_bool(from_int(1))
/// Ok(True)
///
/// import argamak/axis.{Infer}
/// import argamak/space
/// > assert Ok(d1) = space.d1(Infer("X"))
/// > assert Ok(x) = from_ints(of: [1], into: d1)
/// > to_bool(x)
/// Error(IncompatibleShape)
/// ```
///
pub fn to_bool(x: Tensor(a)) -> Result(Bool, TensorError) {
  x
  // Cast to "bool" first as truncation may skew results otherwise
  |> all(with: fn(_) { False })
  |> to_int
  |> result.map(with: int_to_bool)
}

/// Converts a `Tensor` into a flat list of floats.
///
/// ## Examples
///
/// ```gleam
/// > to_floats(from_int(0))
/// [0.]
///
/// > import argamak/axis.{X, Y}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(3), Y(1))
/// > assert Ok(x) = from_floats(of: [1., 2., 3.], into: d2)
/// > to_floats(x)
/// [1., 2., 3.]
/// ```
///
pub fn to_floats(x: Tensor(a)) -> List(Float) {
  x
  |> to_native
  |> do_to_floats
}

if erlang {
  external fn do_to_floats(Native) -> List(Float) =
    "argamak_ffi" "to_floats"
}

if javascript {
  external fn do_to_floats(Native) -> List(Float) =
    "../argamak_ffi.mjs" "to_floats"
}

/// Converts a `Tensor` into a flat list of integers.
///
/// ## Examples
///
/// ```gleam
/// > to_ints(from_float(0.))
/// [0]
///
/// > import argamak/axis.{X, Y}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(3), Y(1))
/// > assert Ok(x) = from_ints(of: [1, 2, 3], into: d2)
/// > to_ints(x)
/// [1, 2, 3]
/// ```
///
pub fn to_ints(x: Tensor(a)) -> List(Int) {
  x
  |> to_native
  |> do_to_ints
}

if erlang {
  external fn do_to_ints(Native) -> List(Int) =
    "argamak_ffi" "to_ints"
}

if javascript {
  external fn do_to_ints(Native) -> List(Int) =
    "../argamak_ffi.mjs" "to_ints"
}

/// Converts a `Tensor` into a flat list of booleans.
///
/// Nonzero values become `True`, otherwise `False`.
///
/// ## Examples
///
/// ```gleam
/// > to_bools(from_float(0.))
/// [False]
///
/// > import argamak/axis.{X, Y}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(3), Y(1))
/// > assert Ok(x) = from_ints(of: [1, 0, -3], into: d2)
/// > to_bools(x)
/// [True, False, True]
/// ```
///
pub fn to_bools(x: Tensor(a)) -> List(Bool) {
  x
  // Cast to "bool" first as truncation may skew results otherwise
  |> all(with: fn(_) { False })
  |> to_ints
  |> list.map(with: int_to_bool)
}

/// Converts a `Tensor` into its `Native` representation.
///
/// ## Examples
///
/// ```gleam
/// > external fn erlang_rank(Native) -> Int =
/// >   "Elixir.Nx" "rank"
/// > to_native(from_int(3)) |> erlang_rank
/// 0
/// ```
///
pub fn to_native(x: Tensor(a)) -> Native {
  x.data
}

pub type ToString {
  Data
  Record
}

/// Returns a string representation of the given `Tensor`, either the whole
/// `Record` or just its `Data`.
///
/// Takes a `column` argument for which the special values `-1` and `0`
/// represent default and no wrapping, respectively.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{X, Y, Z}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d2)
/// > to_string(from: x, return: Data, wrap_at: 0)
/// "[[1, 2],
///  [3, 4]]"
///
/// > assert Ok(d3) = space.d3(X(2), Y(1), Z(4))
/// > let xs = [1, 2, 3, 4, 5, 6, 7, 8]
/// > assert Ok(x) = from_ints(of: xs, into: d3)
/// > do_print(from: x, return: Record, wrap_at: 10)
/// "Tensor(
///   Format(Int32),
///   Space(X(2), Y(1), Z(4)),
///   [[[1, 2,
///      3, 4]],
///    [[5, 6,
///      7, 8]]],
/// )"
/// ```
///
pub fn to_string(
  from x: Tensor(a),
  return record_or_data: ToString,
  wrap_at column: Int,
) -> String {
  let column = case column < 0 {
    True -> columns()
    False -> column
  }
  let tab = case record_or_data {
    Record -> 2
    Data -> 0
  }
  let data =
    x
    |> do_to_string(wrap_at: column, with: tab)
  case record_or_data {
    Record -> {
      let format =
        x
        |> format
        |> format.to_string
      let space =
        x
        |> space
        |> space.to_string
      let space = case string.length(space) > column {
        True if column > 0 ->
          space
          |> string.replace(each: "Space(", with: "Space(\n    ")
          |> string.replace(each: "), ", with: "),\n    ")
          |> string.replace(each: "))", with: "),\n  )")
        _else -> space
      }
      ["Tensor(", "  " <> format <> ",", "  " <> space <> ",", data <> ",", ")"]
      |> string.join(with: "\n")
    }
    Data -> data
  }
}

if erlang {
  external fn columns() -> Int =
    "argamak_ffi" "columns"
}

if javascript {
  external fn columns() -> Int =
    "../argamak_ffi.mjs" "columns"
}

type ToStringAcc {
  ToStringAcc(built: List(StringBuilder), builder: StringBuilder)
}

// fn do_to_string(from x: Tensor(a), wrap_at column: Int, with tab: Int) -> String {
//   let #(xs, item_length) =
//     x
//     |> to_native
//     |> prepare_to_string
// 
//   let rank = rank(x)
// 
//   let shape = case rank {
//     0 -> [1]
//     _else ->
//       x
//       |> shape
//       |> list.reverse
//   }
// 
//   let try_wrap = case column > 0 {
//     True -> {
//       let max_length = column - tab - rank * 2
//       let item_length = item_length + 2
//       let wrap_at = int.max(max_length / item_length, 1)
//       let inner_size = case shape {
//         [inner_size, ..] -> inner_size
//         _else -> 0
//       }
//       fn(j, f) {
//         case { j + 1 } % wrap_at {
//           0 if wrap_at < inner_size -> fn() { Ok(f()) }
//           _else -> fn() { Error(Nil) }
//         }
//       }
//     }
//     False -> fn(_, _) { fn() { Error(Nil) } }
//   }
// 
//   let ToStringAcc(builder: init_builder, ..) as to_string_acc =
//     ToStringAcc(built: [], builder: string_builder.new())
// 
//   let xs =
//     xs
//     |> iterator.from_list
//     |> iterator.map(with: fn(x) {
//       x
//       |> string.pad_left(to: item_length, with: " ")
//       |> string_builder.from_string
//     })
//     |> iterator.index
// 
//   let [#(_, xs)] =
//     shape
//     |> list.index_fold(
//       from: xs,
//       with: fn(acc, size, i) {
//         let try_build = fn(j, f) {
//           case { j + 1 } % size {
//             0 -> Ok(f())
//             _else -> Error(Nil)
//           }
//         }
// 
//         let ToStringAcc(built: built, ..) = case i {
//           0 ->
//             acc
//             |> iterator.fold(
//               from: to_string_acc,
//               with: fn(acc, item) {
//                 let #(j, x) = item
//                 let builder =
//                   string_builder.append_builder(to: acc.builder, suffix: x)
//                 j
//                 |> try_build(fn() {
//                   case rank {
//                     0 ->
//                       ToStringAcc(
//                         ..acc,
//                         built: list.append(acc.built, [builder]),
//                       )
//                     _else -> {
//                       let builder =
//                         builder
//                         |> string_builder.prepend(prefix: "[")
//                         |> string_builder.append(suffix: "]")
//                       ToStringAcc(
//                         built: list.append(acc.built, [builder]),
//                         builder: init_builder,
//                       )
//                     }
//                   }
//                 })
//                 |> result.lazy_or(try_wrap(
//                   j,
//                   fn() {
//                     let indent = string.repeat(" ", times: tab + rank)
//                     let builder =
//                       builder
//                       |> string_builder.append(suffix: ",\n")
//                       |> string_builder.append(suffix: indent)
//                     ToStringAcc(..acc, builder: builder)
//                   },
//                 ))
//                 |> result.lazy_unwrap(or: fn() {
//                   let builder = string_builder.append(to: builder, suffix: ", ")
//                   ToStringAcc(..acc, builder: builder)
//                 })
//               },
//             )
//           _else ->
//             acc
//             |> iterator.fold(
//               from: to_string_acc,
//               with: fn(acc, item) {
//                 let #(j, x) = item
//                 let builder =
//                   string_builder.append_builder(to: acc.builder, suffix: x)
//                 j
//                 |> try_build(fn() {
//                   let builder =
//                     builder
//                     |> string_builder.prepend(prefix: "[")
//                     |> string_builder.append(suffix: "]")
//                   ToStringAcc(
//                     built: list.append(acc.built, [builder]),
//                     builder: init_builder,
//                   )
//                 })
//                 |> result.lazy_unwrap(or: fn() {
//                   let indent = string.repeat(" ", times: tab + rank - i)
//                   let builder =
//                     builder
//                     |> string_builder.append(suffix: ",\n")
//                     |> string_builder.append(suffix: indent)
//                   ToStringAcc(..acc, builder: builder)
//                 })
//               },
//             )
//         }
//         built
//         |> iterator.from_list
//         |> iterator.index
//       },
//     )
//     |> iterator.to_list
//   let indent = string.repeat(" ", times: tab)
//   xs
//   |> string_builder.prepend(prefix: indent)
//   |> string_builder.to_string
// }
fn do_to_string(from x: Tensor(a), wrap_at column: Int, with tab: Int) -> String {
  let #(xs, item_length) =
    x
    |> to_native
    |> prepare_to_string

  let rank = rank(x)

  let shape = case rank {
    0 -> [1]
    _else ->
      x
      |> shape
      |> list.reverse
  }

  let should_wrap = case column > 0 {
    True -> {
      let max_length = column - tab - rank * 2
      let item_length = item_length + 2
      let wrap_at = int.max(max_length / item_length, 1)
      let inner_size = case shape {
        [inner_size, ..] -> inner_size
        _else -> 0
      }
      fn(j) { { j + 1 } % wrap_at == 0 && wrap_at < inner_size }
    }
    False -> fn(_) { False }
  }

  let ToStringAcc(builder: init_builder, ..) as to_string_acc =
    ToStringAcc(built: [], builder: string_builder.new())

  let xs =
    iterator.index({
      use x <- iterator.map(over: iterator.from_list(xs))
      x
      |> string.pad_left(to: item_length, with: " ")
      |> string_builder.from_string
    })

  let [#(_, xs)] =
    iterator.to_list({
      use acc, size, i <- list.index_fold(over: shape, from: xs)
      let should_build = fn(j) { { j + 1 } % size == 0 }
      let ToStringAcc(built: built, ..) = case i {
        0 -> {
          use acc, item <- iterator.fold(over: acc, from: to_string_acc)
          let #(j, x) = item
          let builder =
            string_builder.append_builder(to: acc.builder, suffix: x)
          let should_build_j = should_build(j)
          use <- when(
              should_build_j && rank == 0,
              then: fn() {
                ToStringAcc(..acc, built: list.append(acc.built, [builder]))
              },
            )
          use <- when(
              should_build_j,
              then: fn() {
                let builder =
                  builder
                  |> string_builder.prepend(prefix: "[")
                  |> string_builder.append(suffix: "]")
                ToStringAcc(
                  built: list.append(acc.built, [builder]),
                  builder: init_builder,
                )
              },
            )
          use <- when(
              should_wrap(j),
              then: fn() {
                let indent = string.repeat(" ", times: tab + rank)
                let builder =
                  builder
                  |> string_builder.append(suffix: ",\n")
                  |> string_builder.append(suffix: indent)
                ToStringAcc(..acc, builder: builder)
              },
            )
          // else
          let builder = string_builder.append(to: builder, suffix: ", ")
          ToStringAcc(..acc, builder: builder)
        }
        _else -> {
          use acc, item <- iterator.fold(over: acc, from: to_string_acc)
          let #(j, x) = item
          let builder =
            string_builder.append_builder(to: acc.builder, suffix: x)
          use <- when(
              should_build(j),
              then: fn() {
                let builder =
                  builder
                  |> string_builder.prepend(prefix: "[")
                  |> string_builder.append(suffix: "]")
                ToStringAcc(
                  built: list.append(acc.built, [builder]),
                  builder: init_builder,
                )
              },
            )
          // else
          let indent = string.repeat(" ", times: tab + rank - i)
          let builder =
            builder
            |> string_builder.append(suffix: ",\n")
            |> string_builder.append(suffix: indent)
          ToStringAcc(..acc, builder: builder)
        }
      }
      built
      |> iterator.from_list
      |> iterator.index
    })

  let indent = string.repeat(" ", times: tab)
  xs
  |> string_builder.prepend(prefix: indent)
  |> string_builder.to_string
}

if erlang {
  external fn prepare_to_string(Native) -> #(List(String), Int) =
    "argamak_ffi" "prepare_to_string"
}

if javascript {
  external fn prepare_to_string(Native) -> #(List(String), Int) =
    "../argamak_ffi.mjs" "prepare_to_string"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Utility Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Prints the data and metadata from a given `Tensor` and returns the `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{X, Y, Z}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d2)
/// > debug(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 2],
///    [3, 4]],
/// )
/// x
///
/// > assert Ok(d3) = space.d3(X(2), Y(1), Z(4))
/// > let xs = [1, 2, 3, 4, 5, 6, 7, 8]
/// > assert Ok(x) = from_ints(of: xs, into: d3)
/// > debug(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(1), Z(4)),
///   [[[1, 2, 3, 4]],
///    [[5, 6, 7, 8]]],
/// )
/// x
/// ```
///
pub fn debug(x: Tensor(a)) -> Tensor(a) {
  print(x)
  x
}

/// Prints the data and metadata from a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{X, Y, Z}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d2)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(2)),
///   [[1, 2],
///    [3, 4]],
/// )
/// Nil
///
/// > assert Ok(d3) = space.d3(X(2), Y(1), Z(4))
/// > let xs = [1, 2, 3, 4, 5, 6, 7, 8]
/// > assert Ok(x) = from_ints(of: xs, into: d3)
/// > print(x)
/// Tensor(
///   Format(Int32),
///   Space(X(2), Y(1), Z(4)),
///   [[[1, 2, 3, 4]],
///    [[5, 6, 7, 8]]],
/// )
/// Nil
/// ```
///
pub fn print(x: Tensor(a)) -> Nil {
  x
  |> to_string(return: Record, wrap_at: -1)
  |> io.println
}

/// Prints the data from a given `Tensor`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{X, Y, Z}
/// > import argamak/space
/// > assert Ok(d2) = space.d2(X(2), Y(2))
/// > assert Ok(x) = from_ints(of: [1, 2, 3, 4], into: d2)
/// > print_data(x)
/// [[1, 2],
///  [3, 4]]
/// Nil
///
/// > assert Ok(d3) = space.d3(X(2), Y(1), Z(4))
/// > let xs = [1, 2, 3, 4, 5, 6, 7, 8]
/// > assert Ok(x) = from_ints(of: xs, into: d3)
/// > print_data(x)
/// [[[1, 2, 3, 4]],
///  [[5, 6, 7, 8]]]
/// Nil
/// ```
///
pub fn print_data(x: Tensor(a)) -> Nil {
  x
  |> to_string(return: Data, wrap_at: -1)
  |> io.println
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Private Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a `Tensor` put into a given `Space` with a `Format` applied to
/// the data on success, or a `TensorError` on failure.
///
fn tensor(
  from data: a,
  into space: Space,
  with new_format: Format(b),
) -> TensorResult(b) {
  try native = do_tensor(data, format.to_native(new_format))
  try x =
    native
    |> Tensor(format: new_format, space: space)
    |> reshape(into: space)
  Ok(x)
}

if erlang {
  external fn do_tensor(data, format) -> NativeResult =
    "argamak_ffi" "tensor"
}

if javascript {
  external fn do_tensor(data, format) -> NativeResult =
    "../argamak_ffi.mjs" "tensor"
}

type FitBy {
  Definition
  Inference
}

type FitAcc(axis) {
  FitAcc(divisor: Int, fit_by: FitBy)
}

/// Results in the given `Tensor` on success, or a `TensorError` on failure.
///
/// Ensures the `Tensor` is compatible with its `Space` and converts a maximum
/// of one `Infer` into an `Axis` of known `size`.
///
/// Useful in low-level functions where a `Tensor` is put into a new `Space`.
///
fn fit(x: Tensor(a)) -> TensorResult(a) {
  let dividend = size(x)
  // let FitAcc(divisor: divisor, fit_by: fit_by) =
  //   x
  //   |> axes
  //   |> list.fold(
  //     from: FitAcc(divisor: 1, fit_by: Definition),
  //     with: fn(acc, axis) {
  //       case axis {
  //         Infer(_) -> FitAcc(..acc, fit_by: Inference)
  //         _else -> FitAcc(..acc, divisor: acc.divisor * axis.size(axis))
  //       }
  //     },
  //   )
  let FitAcc(divisor: divisor, fit_by: fit_by) = {
    use
      acc,
      axis
    <- list.fold(over: axes(x), from: FitAcc(divisor: 1, fit_by: Definition))
    case axis {
      Infer(_) -> FitAcc(..acc, fit_by: Inference)
      _else -> FitAcc(..acc, divisor: acc.divisor * axis.size(axis))
    }
  }

  case dividend % divisor {
    0 if fit_by == Definition -> Ok(x)
    0 if fit_by == Inference -> {
      // assert Ok(space) =
      //   x
      //   |> space
      //   |> space.map(with: fn(axis) {
      //     case axis {
      //       Infer(_) -> axis.resize(axis, dividend / divisor)
      //       _else -> axis
      //     }
      //   })
      assert Ok(space) = {
        use axis <- space.map(space(x))
        case axis {
          Infer(_) -> axis.resize(axis, dividend / divisor)
          _else -> axis
        }
      }
      Tensor(..x, space: space)
      |> Ok
    }
    _else -> Error(IncompatibleShape)
  }
}

fn broadcastable(
  f: fn(Native, Native) -> NativeResult,
  a: Tensor(a),
  b: Tensor(a),
) -> TensorResult(a) {
  try space =
    a
    |> space
    |> space.merge(space(b))
    |> result.map_error(with: SpaceErrors)
  try native = f(to_native(a), to_native(b))
  Tensor(..a, data: native, space: space)
  |> Ok
}

fn sign_not_equal(a: Tensor(a), b: Tensor(a)) -> TensorResult(a) {
  let zero =
    0
    |> from_int
    |> reformat(apply: format(a))
  try x = multiply(sign(a), sign(b))
  less(is: x, than: zero)
}

fn permit_zero(
  in x: Tensor(a),
  with f: fn(Tensor(a)) -> TensorResult(a),
) -> TensorResult(a) {
  let zero =
    0
    |> from_int
    |> reformat(apply: format(x))
  try is_nonzero = not_equal(is: x, to: zero)
  try is_zero = equal(is: x, to: zero)
  try x = add(x, is_zero)
  try x = f(x)
  multiply(x, is_nonzero)
}

fn all_nonzero(x: Tensor(a)) -> TensorResult(a) {
  try all =
    x
    |> all(with: fn(_axis) { True })
    |> reformat(apply: format.int32())
    |> to_int
  case all {
    1 -> Ok(x)
    _else -> Error(ZeroDivision)
  }
}

type Reducible {
  Away
  InSitu
}

type ReducibleAcc {
  ReducibleAcc(axes: Axes, indices: Indices)
}

fn reducible_over_axes(
  f: fn(Native, Indices) -> Native,
  x: Tensor(a),
  filter: fn(Axis) -> Bool,
  reduce: Reducible,
) -> Tensor(a) {
  // let acc =
  //   x
  //   |> axes
  //   |> list.index_fold(
  //     from: ReducibleAcc([], []),
  //     with: fn(acc, axis, index) {
  //       case filter(axis) {
  //         True if reduce == InSitu -> {
  //           let axis = axis.resize(axis, 1)
  //           ReducibleAcc(
  //             axes: [axis, ..acc.axes],
  //             indices: [index, ..acc.indices],
  //           )
  //         }
  //         True -> ReducibleAcc(..acc, indices: [index, ..acc.indices])
  //         False -> ReducibleAcc(..acc, axes: [axis, ..acc.axes])
  //       }
  //     },
  //   )
  let acc = {
    use
      acc,
      axis,
      index
    <- list.index_fold(over: axes(x), from: ReducibleAcc([], []))
    case filter(axis) {
      True if reduce == InSitu -> {
        let axis = axis.resize(axis, 1)
        ReducibleAcc(axes: [axis, ..acc.axes], indices: [index, ..acc.indices])
      }
      True -> ReducibleAcc(..acc, indices: [index, ..acc.indices])
      False -> ReducibleAcc(..acc, axes: [axis, ..acc.axes])
    }
  }
  assert Ok(new_space) =
    acc.axes
    |> list.reverse
    |> space.from_list
  let native = to_native(x)
  let native = case list.reverse(acc.indices) {
    [] -> {
      // Normalize by appending a size-1 axis to reduce over (TensorFlow-like)
      assert Ok(native) =
        x
        |> shape
        |> list.append([1])
        |> do_reshape(native, _)
      f(native, [rank(x)])
    }
    indices -> {
      assert Ok(native) =
        native
        |> f(indices)
        |> do_reshape(space.shape(new_space))
      native
    }
  }
  Tensor(..x, data: native, space: new_space)
}

fn reducible_over_axis(
  f: fn(Native, Int) -> Native,
  x: Tensor(a),
  find: fn(Axis) -> Bool,
  reduce: Reducible,
) -> Tensor(a) {
  // let acc =
  //   x
  //   |> axes
  //   |> list.index_fold(
  //     from: ReducibleAcc([], []),
  //     with: fn(acc, axis, index) {
  //       case acc.indices == [] && find(axis) {
  //         True if reduce == InSitu -> {
  //           let axis = axis.resize(axis, 1)
  //           ReducibleAcc(axes: [axis, ..acc.axes], indices: [index])
  //         }
  //         True -> ReducibleAcc(..acc, indices: [index])
  //         False -> ReducibleAcc(..acc, axes: [axis, ..acc.axes])
  //       }
  //     },
  //   )
  let acc = {
    use
      acc,
      axis,
      index
    <- list.index_fold(over: axes(x), from: ReducibleAcc([], []))
    case acc.indices == [] && find(axis) {
      True if reduce == InSitu -> {
        let axis = axis.resize(axis, 1)
        ReducibleAcc(axes: [axis, ..acc.axes], indices: [index])
      }
      True -> ReducibleAcc(..acc, indices: [index])
      False -> ReducibleAcc(..acc, axes: [axis, ..acc.axes])
    }
  }
  assert Ok(new_space) =
    acc.axes
    |> list.reverse
    |> space.from_list
  case acc.indices {
    [] -> {
      // Normalize by flattening the tensor if no index is found (Nx-like)
      assert Ok(new_space) =
        "Nil"
        |> Infer
        |> space.d1
      assert Ok(x) = reshape(put: x, into: new_space)
      Tensor(..x, data: f(to_native(x), 0), space: space.new())
    }
    [index, ..] -> {
      assert Ok(native) =
        x
        |> to_native
        |> f(index)
        |> do_reshape(space.shape(new_space))
      Tensor(..x, data: native, space: new_space)
    }
  }
}

fn int_to_bool(x) {
  case x {
    0 -> False
    _else -> True
  }
}

fn when(condition: Bool, then f: fn() -> a, else g: fn() -> a) -> a {
  case condition {
    True -> f()
    False -> g()
  }
}
