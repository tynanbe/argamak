import gleam/dict
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import argamak/axis.{type Axes, type Axis, Axis, Infer}

/// An n-dimensional `Space` containing `Axes` of various sizes.
///
pub opaque type Space {
  Space(axes: Axes)
}

/// An error returned when attempting to create an invalid `Space`.
///
pub type SpaceError {
  CannotMerge
  CannotInfer
  DuplicateName
  InvalidSize
  SpaceError(reason: SpaceError, axes: Axes)
}

/// A `SpaceError` list.
///
pub type SpaceErrors =
  List(SpaceError)

/// A `Result` alias type for spaces.
///
pub type SpaceResult =
  Result(Space, SpaceErrors)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Creation Functions                     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a dimensionless `Space`.
///
/// ## Examples
///
/// ```gleam
/// > new() |> axes
/// []
/// ```
///
pub fn new() -> Space {
  Space(axes: [])
}

/// Results in a one-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer}
/// > let assert Ok(space) = d1(Infer("A"))
/// > axes(space)
/// [Infer("A")]
/// ```
///
pub fn d1(a: Axis) -> SpaceResult {
  [a]
  |> Space
  |> validate
}

/// Results in a two-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B}
/// > let assert Ok(space) = d2(A(2), B(2))
/// > axes(space)
/// [A(2), B(2)]
/// ```
///
pub fn d2(a: Axis, b: Axis) -> SpaceResult {
  [a, b]
  |> Space
  |> validate
}

/// Results in a three-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B, Infer}
/// > let assert Ok(space) = d3(A(2), B(2), Infer("C"))
/// > axes(space)
/// [A(2), B(2), Infer("C")]
/// ```
///
pub fn d3(a: Axis, b: Axis, c: Axis) -> SpaceResult {
  [a, b, c]
  |> Space
  |> validate
}

/// Results in a four-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B, D, Infer}
/// > let assert Ok(space) = d4(A(2), B(2), Infer("C"), D(1))
/// > axes(space)
/// [A(2), B(2), Infer("C"), D(1)]
/// ```
///
pub fn d4(a: Axis, b: Axis, c: Axis, d: Axis) -> SpaceResult {
  [a, b, c, d]
  |> Space
  |> validate
}

/// Results in a five-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B, C, D, E}
/// > let assert Ok(space) = d5(A(5), B(4), C(3), D(2), E(1))
/// > axes(space)
/// [A(5), B(4), C(3), D(2), E(1)]
/// ```
///
pub fn d5(a: Axis, b: Axis, c: Axis, d: Axis, e: Axis) -> SpaceResult {
  [a, b, c, d, e]
  |> Space
  |> validate
}

/// Results in a six-dimensional `Space` on success, or `SpaceErrors` on
/// failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B, C, D, E, F}
/// > let assert Ok(space) = d6(A(9), B(9), C(9), D(9), E(9), F(9))
/// > axes(space)
/// [A(9), B(9), C(9), D(9), E(9), F(9)]
/// ```
///
pub fn d6(a: Axis, b: Axis, c: Axis, d: Axis, e: Axis, f: Axis) -> SpaceResult {
  [a, b, c, d, e, f]
  |> Space
  |> validate
}

/// Results in a `Space` created from a list of `Axes` on success, or
/// `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{A, B, C, D, E, F, Z}
/// > let assert Ok(space) = from_list([A(9), B(9), C(9), D(9), E(9), F(9), Z(9)])
/// > axes(space)
/// [A(9), B(9), C(9), D(9), E(9), F(9), Z(9)]
/// ```
///
pub fn from_list(x: Axes) -> SpaceResult {
  x
  |> Space
  |> validate
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reflection Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Returns the axes of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > new() |> axes
/// []
///
/// > import argamak/axis.{A, B, Infer}
/// > let assert Ok(space) = d1(Infer("A"))
/// > axes(space)
/// [Infer("A")]
///
/// > let assert Ok(space) = d3(A(2), B(2), Infer("C"))
/// > axes(space)
/// [A(2), B(2), Infer("C")]
/// ```
///
pub fn axes(x: Space) -> Axes {
  x.axes
}

/// Returns the degree of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > new() |> degree
/// 0
///
/// > import argamak/axis.{A, B, Infer}
/// > let assert Ok(space) = d1(Infer("A"))
/// > degree(space)
/// 1
///
/// > let assert Ok(space) = d3(A(2), B(2), Infer("C"))
/// > degree(space)
/// 3
/// ```
///
pub fn degree(x: Space) -> Int {
  x
  |> axes
  |> list.length
}

/// Returns the shape of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > new() |> shape
/// []
///
/// > import argamak/axis.{A, B, Infer}
/// > let assert Ok(space) = d1(Infer("A"))
/// > shape(space)
/// [0]
///
/// > let assert Ok(space) = d3(A(2), B(2), Infer("C"))
/// > shape(space)
/// [2, 2, 0]
/// ```
///
pub fn shape(x: Space) -> List(Int) {
  x
  |> axes
  |> list.map(with: axis.size)
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Transformation Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in a new `Space` with the same number of dimensions as the given
/// `Space` on success, or `SpaceErrors` on failure.
///
/// Applies the given function to each `Axis` of the `Space`.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{B, C, Infer}
/// > let assert Ok(space) = map(new(), with: fn(_) { C(3) })
/// > axes(space)
/// []
///
/// > let assert Ok(space) = d1(Infer("A"))
/// > let assert Ok(space) = map(space, with: fn(_) { C(3) })
/// > axes(space)
/// [C(3)]
///
/// > let assert Ok(space) = d3(Infer("A"), B(2), C(2))
/// > let assert Ok(space) = map(space, with: fn(axis) {
/// >   case axis {
/// >     Infer(_) -> axis.resize(axis, 4)
/// >     _else -> axis
/// >   }
/// > })
/// > axes(space)
/// [A(4), B(2), C(2)]
/// ```
///
pub fn map(x: Space, with fun: fn(Axis) -> Axis) -> SpaceResult {
  x
  |> axes
  |> list.map(with: fun)
  |> Space
  |> validate
}

/// Results in a new `Space` that is the element-wise maximum of the given
/// spaces on success, or `SpaceErrors` on failure.
///
/// Spaces are merged tail-first, and corresponding `Axis` names must match.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Axis, Infer, X, Y}
/// > let assert Ok(a) = d1(Infer("X"))
/// > merge(a, new()) |> result.map(with: axes)
/// Ok([Infer("X")])
///
/// > let assert Ok(b) = d2(Axis("Sparkle", 2), X(2))
/// > merge(a, b) |> result.map(with: axes)
/// Ok([Axis("Sparkle", 2), Infer("X")])
///
/// > let assert Ok(c) = d3(Infer("X"), Axis("Sparkle", 3), Y(3))
/// > merge(b, c)
/// Error([SpaceError(CannotMerge, [Y(3), X(2)])])
/// ```
///
pub fn merge(a: Space, b: Space) -> SpaceResult {
  let index = fn(x: Space) {
    x
    |> axes
    |> list.index_map(with: fn(axis, index) { #(index, axis) })
    |> dict.from_list
  }
  let a_index = index(a)
  let b_index = index(b)

  let a_size = dict.size(a_index)
  let b_size = dict.size(b_index)

  let #(x, dict) = case a_size < b_size {
    True -> #(axes(b), a_index)
    False -> #(axes(a), b_index)
  }
  let offset = int.absolute_value(a_size - b_size)

  let #(x, errors) =
    x
    |> list.index_map(with: fn(a_axis, index) {
      let b_axis =
        dict
        |> dict.get(index - offset)
        |> result.unwrap(or: a_axis)
      let a_name = axis.name(a_axis)
      let b_name = axis.name(b_axis)
      let a_size = axis.size(a_axis)
      let b_size = axis.size(b_axis)
      let should_infer = a_axis == Infer(a_name) || b_axis == Infer(b_name)
      case a_name == b_name {
        True if should_infer ->
          a_name
          |> Infer
          |> Ok
        True ->
          a_axis
          |> axis.resize(int.max(a_size, b_size))
          |> Ok
        False ->
          CannotMerge
          |> SpaceError(axes: [a_axis, b_axis])
          |> Error
      }
    })
    |> list.partition(with: result.is_ok)

  use x <- result.try(case errors {
    [] ->
      x
      |> result.all
      |> result.map_error(with: fn(error) { [error] })
    _else ->
      errors
      |> list.map(with: fn(x) {
        let assert Error(x) = x
        x
      })
      |> Error
  })

  x
  |> Space
  |> validate
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Conversion Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Converts a `Space` into a `String`.
///
/// ## Examples
///
/// ```gleam
/// > new() |> to_string
/// "Space()"
///
/// > import argamak/axis.{A, B, Axis, Infer}
/// > let assert Ok(space) = d1(Axis("Sparkle", 2))
/// > to_string(space)
/// "Space(Axis(\"Sparkle\", 2))"
///
/// > let assert Ok(space) = d3(A(2), B(2), Infer("C"))
/// > to_string(space)
/// "Space(A(2), B(2), Infer(\"C\"))"
/// ```
///
pub fn to_string(x: Space) -> String {
  let axes = {
    use x <- list.map(axes(x))
    let name = axis.name(x)
    let size =
      x
      |> axis.size
      |> int.to_string
    case x {
      Axis(..) -> "Axis(\"" <> name <> "\", " <> size <> ")"
      Infer(_) -> "Infer(\"" <> name <> "\")"
      _else -> name <> "(" <> size <> ")"
    }
  }
  "Space(" <> string.join(axes, with: ", ") <> ")"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Private Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Results in the given `Space` on success, or `SpaceErrors` on failure.
///
/// Ensures that no axes are duplicated, that there is at most a single
/// inferred dimension size, and that no other dimension sizes are less than
/// one.
///
/// ## Examples
///
/// ```gleam
/// > import argamak/axis.{Infer, X, Y, Z}
/// > validate(d3(X(1), Infer("Y"), Z(1)))
/// Ok(space)
///
/// > validate(d2(X(1), Infer("X")))
/// Error([SpaceError(DuplicateName, [Infer("X")])])
///
/// > validate(d2(Infer("X"), Infer("Y")))
/// Error([SpaceError(CannotInfer, [Infer("Y")])])
///
/// > validate(d2(X(0), Y(1)))
/// Error([SpaceError(InvalidSize, [X(0)])])
///
/// > validate(d3(X(-2), Infer("X"), Infer("Z")))
/// Error([
///   SpaceError(InvalidSize, [X(-2)]),
///   SpaceError(DuplicateName, [Infer("X")]),
///   SpaceError(CannotInfer, [Infer("Z")]),
/// ])
/// ```
///
fn validate(space: Space) -> SpaceResult {
  let ValidateAcc(_, _, results: results) = {
    use acc, axis <- list.fold(
      over: axes(space),
      from: ValidateAcc(names: [], inferred: False, results: []),
    )
    let name = axis.name(axis)
    let size = axis.size(axis)
    let errors =
      [
        Invalid(error: DuplicateName, when: list.contains(acc.names, any: name)),
        Invalid(error: CannotInfer, when: acc.inferred && axis == Infer(name)),
        Invalid(error: InvalidSize, when: size < 1 && axis != Infer(name)),
      ]
      |> list.map(with: fn(invalid) {
        case invalid.when {
          False -> []
          True -> [SpaceError(reason: invalid.error, axes: [axis])]
        }
      })
      |> list.flatten
    let result = case errors {
      [] -> Ok(axis)
      _else -> Error(errors)
    }
    ValidateAcc(
      names: [name, ..acc.names],
      inferred: acc.inferred || axis == Infer(name),
      results: [result, ..acc.results],
    )
  }

  case list.any(in: results, satisfying: result.is_error) {
    False -> Ok(space)
    True ->
      results
      |> list.reverse
      |> list.map(with: fn(result) {
        case result {
          Ok(_) -> []
          Error(errors) -> errors
        }
      })
      |> list.flatten
      |> Error
  }
}

type ValidateAcc {
  ValidateAcc(
    names: List(String),
    inferred: Bool,
    results: List(Result(Axis, SpaceErrors)),
  )
}

type Invalid {
  Invalid(error: SpaceError, when: Bool)
}
