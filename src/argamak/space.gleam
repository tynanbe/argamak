import argamak/util
import gleam/int
import gleam/list
import gleam/pair
import gleam/result
import gleam/string

/// An n-dimensional `Space` with axis–shape tuple elements, one per dimension.
///
pub opaque type Space(dn, axis) {
  Space(degree: dn, elements: List(#(axis, Int)))
}

/// An error returned when attempting to create an invalid `Space`.
///
pub type SpaceError {
  SpaceError(error: String, element: String)
}

/// A `SpaceError` list.
///
pub type SpaceErrors =
  List(SpaceError)

/// A type without dimensions.
///
pub type D0 {
  D0
}

/// A type with one dimension.
///
pub type D1 {
  D1
}

/// A type with two dimensions.
///
pub type D2 {
  D2
}

/// A type with three dimensions.
///
pub type D3 {
  D3
}

/// A type with four dimensions.
///
pub type D4 {
  D4
}

/// A type with five dimensions.
///
pub type D5 {
  D5
}

/// A type with six dimensions.
///
pub type D6 {
  D6
}

/// Results in a dimensionless `Space`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > elements(space)
/// []
/// ```
///
pub fn d0() -> Result(Space(D0, axis), SpaceErrors) {
  Space(degree: D0, elements: [])
  |> Ok
}

/// Results in a 1-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A }
/// > assert Ok(space) = d1(#(A, -1))
/// > elements(space)
/// [#(A, -1)]
/// ```
///
pub fn d1(a: #(axis, Int)) -> Result(Space(D1, axis), SpaceErrors) {
  Space(degree: D1, elements: [a])
  |> validate
}

/// Results in a 2-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B }
/// > assert Ok(space) = d2(#(A, 2), #(B, 2))
/// > elements(space)
/// [#(A, 2), #(B, 2)]
/// ```
///
pub fn d2(
  a: #(axis, Int),
  b: #(axis, Int),
) -> Result(Space(D2, axis), SpaceErrors) {
  Space(degree: D2, elements: [a, b])
  |> validate
}

/// Results in a 3-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B C }
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > elements(space)
/// [#(A, 2), #(B, 2), #(C, -1)]
/// ```
///
pub fn d3(
  a: #(axis, Int),
  b: #(axis, Int),
  c: #(axis, Int),
) -> Result(Space(D3, axis), SpaceErrors) {
  Space(degree: D3, elements: [a, b, c])
  |> validate
}

/// Results in a 4-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B C D }
/// > assert Ok(space) = d4(#(A, 2), #(B, 2), #(C, -1), #(D, 1))
/// > elements(space)
/// [#(A, 2), #(B, 2), #(C, -1), #(D, 1)]
/// ```
///
pub fn d4(
  a: #(axis, Int),
  b: #(axis, Int),
  c: #(axis, Int),
  d: #(axis, Int),
) -> Result(Space(D4, axis), SpaceErrors) {
  Space(degree: D4, elements: [a, b, c, d])
  |> validate
}

/// Results in a 5-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B C D E }
/// > assert Ok(space) = d5(#(A, 5), #(B, 4), #(C, 3), #(D, 2), #(E, 1))
/// > elements(space)
/// [#(A, 5), #(B, 4), #(C, 3), #(D, 2), #(E, 1)]
/// ```
///
pub fn d5(
  a: #(axis, Int),
  b: #(axis, Int),
  c: #(axis, Int),
  d: #(axis, Int),
  e: #(axis, Int),
) -> Result(Space(D5, axis), SpaceErrors) {
  Space(degree: D5, elements: [a, b, c, d, e])
  |> validate
}

/// Results in a 6-dimensional `Space` on success, or `SpaceErrors` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B C D E F }
/// > assert Ok(space) =
/// >   d6(#(A, 9), #(B, 9), #(C, 9), #(D, 9), #(E, 9), #(F, 9))
/// > elements(space)
/// [#(A, 9), #(B, 9), #(C, 9), #(D, 9), #(E, 9), #(F, 9)]
/// ```
///
pub fn d6(
  a: #(axis, Int),
  b: #(axis, Int),
  c: #(axis, Int),
  d: #(axis, Int),
  e: #(axis, Int),
  f: #(axis, Int),
) -> Result(Space(D6, axis), SpaceErrors) {
  Space(degree: D6, elements: [a, b, c, d, e, f])
  |> validate
}

/// Returns the axes of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > axes(space)
/// []
///
/// > type Axis { A B C }
/// > assert Ok(space) = d1(#(A, -1))
/// > axes(space)
/// [A]
///
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > axes(space)
/// [A, B, C]
/// ```
///
pub fn axes(of space: Space(dn, axis)) -> List(axis) {
  space
  |> elements
  |> list.map(with: pair.first)
}

/// Returns the degree of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > degree(space)
/// D0
///
/// > type Axis { A B C }
/// > assert Ok(space) = d1(#(A, -1))
/// > degree(space)
/// D1
///
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > degree(space)
/// D3
/// ```
///
pub fn degree(of space: Space(dn, axis)) -> dn {
  space.degree
}

/// Returns the elements of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > elements(space)
/// []
///
/// > type Axis { A B C }
/// > assert Ok(space) = d1(#(A, -1))
/// > elements(space)
/// [#(A, -1)]
///
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > elements(space)
/// [#(A, 2), #(B, 2), #(C, -1)]
/// ```
///
pub fn elements(of space: Space(dn, axis)) -> List(#(axis, Int)) {
  space.elements
}

/// Returns the shape of a given `Space`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > shape(space)
/// []
///
/// > type Axis { A B C }
/// > assert Ok(space) = d1(#(A, -1))
/// > shape(space)
/// [-1]
///
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > shape(space)
/// [2, 2, -1]
/// ```
///
pub fn shape(of space: Space(dn, axis)) -> List(Int) {
  space
  |> elements
  |> list.map(with: pair.second)
}

/// Results in a new `Space` with the same number of dimensions as the given
/// `Space` on success, or `SpaceErrors` on failure.
///
/// Applies the given function to each element of the `Space`.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { A B C }
/// > assert Ok(space) = d0()
/// > assert Ok(space) = map_elements(of: space, with: fn(_) { #(C, 3) })
/// > elements(space)
/// []
///
/// > assert Ok(space) = d1(#(A, -1))
/// > assert Ok(space) = map_elements(of: space, with: fn(_) { #(C, 3) })
/// > elements(space)
/// [#(C, 3)]
///
/// > assert Ok(space) = d3(#(A, -1), #(B, 2), #(C, 2))
/// > assert Ok(space) = map_elements(of: space, with: fn(element) {
/// >   let #(axis, size) = element
/// >   case size == -1 {
/// >     True -> #(axis, 4)
/// >     False -> element
/// >   }
/// > })
/// > elements(space)
/// [#(A, 4), #(B, 2), #(C, 2)]
/// ```
///
pub fn map_elements(
  of space: Space(dn, a),
  with fun: fn(#(a, Int)) -> #(b, Int),
) -> Result(Space(dn, b), SpaceErrors) {
  Space(
    degree: degree(space),
    elements: space
    |> elements
    |> list.map(with: fun),
  )
  |> validate
}

/// Converts a `Space` into a `String`.
///
/// ## Examples
///
/// ```gleam
/// > assert Ok(space) = d0()
/// > to_string(space)
/// "D0"
///
/// > type Axis { A B C }
/// > assert Ok(space) = d1(#(A, -1))
/// > to_string(space)
/// "D1 #(A, -1)"
///
/// > assert Ok(space) = d3(#(A, 2), #(B, 2), #(C, -1))
/// > to_string(space)
/// "D3 #(A, 2), #(B, 2), #(C, -1)"
/// ```
///
pub fn to_string(space: Space(dn, axis)) -> String {
  let elements =
    space
    |> elements
    |> list.map(with: element_to_string)
    |> string.join(with: ", ")
  let elements = case elements != "" {
    True -> string.append(to: " ", suffix: elements)
    False -> elements
  }

  assert Ok(degree) =
    space
    |> degree
    |> util.record_to_string

  string.append(to: degree, suffix: elements)
}

fn element_to_string(element: #(axis, Int)) -> String {
  assert Ok(axis) =
    element
    |> pair.first
    |> util.record_to_string
  let size =
    element
    |> pair.second
    |> int.to_string
  string.concat(["#(", axis, ", ", size, ")"])
}

/// Results in the given `Space` on success, or `SpaceErrors` on failure.
///
/// Ensures that no axes are duplicated, that there is at most a single
/// inferred dimension size, and that no other dimension sizes are less than
/// one.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { X Y Z }
/// > validate(d3(#(X, 1), #(Y, -1), #(Z, 1)))
/// Ok(space)
///
/// > validate(d2(#(X, 1), #(X, -1)))
/// Error([
///   SpaceError(
///     error: "multiple axis records from same constructor",
///     element: "#(X, -1)"),
/// ])
///
/// > validate(d2(#(X, -1), #(Y, -1)))
/// Error([
///   SpaceError(
///     error: "multiple inferred dimension sizes",
///     element: "#(Y, -1)"),
/// ])
///
/// > validate(d2(#(X, 0), #(Y, 1)))
/// Error([SpaceError(error: "dimension size < 1", element: "#(X, 0)")])
///
/// > validate(d3(#(X, -2), #(X, -1), #(Z, -1)))
/// Error([
///   SpaceError(error: "dimension size < 1", element: "#(X, -2)"),
///   SpaceError(
///     error: "multiple axis records from same constructor",
///     element: "#(X, -1)",
///   ),
///   SpaceError(
///     error: "multiple inferred dimension sizes",
///     element: "#(Z, -1)",
///   ),
/// ])
/// ```
///
fn validate(space: Space(dn, axis)) -> Result(Space(dn, axis), SpaceErrors) {
  let initial = ValidateAcc(axes: [], inferred: False, results: [])
  let ValidateAcc(_, _, results: results) =
    space
    |> elements
    |> list.fold(
      from: initial,
      with: fn(acc: ValidateAcc(axis), element) {
        let #(axis, size) = element
        let errors =
          [
            Invalid(
              message: "multiple axis records from same constructor",
              when: list.contains(acc.axes, any: axis),
            ),
            Invalid(
              message: "multiple inferred dimension sizes",
              when: acc.inferred && size == -1,
            ),
            Invalid(message: "dimension size < 1", when: size < 1 && size != -1),
          ]
          |> list.map(with: fn(invalid: Invalid) {
            case invalid.when {
              False -> []
              True -> [
                SpaceError(
                  error: invalid.message,
                  element: element_to_string(element),
                ),
              ]
            }
          })
          |> list.flatten
        let result = case errors {
          [] -> Ok(element)
          _else -> Error(errors)
        }
        ValidateAcc(
          axes: [axis, ..acc.axes],
          inferred: acc.inferred || size == -1,
          results: [result, ..acc.results],
        )
      },
    )

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

type ValidateAcc(axis) {
  ValidateAcc(
    axes: List(axis),
    inferred: Bool,
    results: List(Result(#(axis, Int), SpaceErrors)),
  )
}

type Invalid {
  Invalid(message: String, when: Bool)
}
