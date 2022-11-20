import gleam/result
import gleam/string

/// The elements that comprise a `Space`.
///
/// Except for `Infer`, every `Axis` has a `size` corresponding to the number of
/// values that fit along that `Axis` when a `Tensor` is put into a `Space`
/// containing that `Axis`.
///
/// An `Axis` can be given a unique `name` using the `Axis` or `Infer`
/// constructors. Single-letter constructors are also provided for convenience.
///
/// The special `Infer` constructor can be used once per `Space`. It will be
/// replaced and have its `size` computed when a `Tensor` is put into that
/// `Space`.
///
pub type Axis {
  Axis(name: String, size: Int)
  Infer(name: String)
  A(size: Int)
  B(size: Int)
  C(size: Int)
  D(size: Int)
  E(size: Int)
  F(size: Int)
  G(size: Int)
  H(size: Int)
  I(size: Int)
  J(size: Int)
  K(size: Int)
  L(size: Int)
  M(size: Int)
  N(size: Int)
  O(size: Int)
  P(size: Int)
  Q(size: Int)
  R(size: Int)
  S(size: Int)
  T(size: Int)
  U(size: Int)
  V(size: Int)
  W(size: Int)
  X(size: Int)
  Y(size: Int)
  Z(size: Int)
}

/// An `Axis` list.
///
pub type Axes =
  List(Axis)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reflection Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Returns the name of a given `Axis`.
///
/// ## Examples
///
/// ```gleam
/// > name(A(1))
/// "A"
///
/// > name(Axis(name: "Sparkle", size: 99))
/// "Sparkle"
///
/// > name(Infer("Silver"))
/// "Silver"
/// ```
pub fn name(x: Axis) -> String {
  case x {
    Axis(name: name, ..) | Infer(name: name) -> name
    _else ->
      x
      |> string.inspect
      |> string.first
      |> result.unwrap(or: "")
  }
}

/// Returns the size of a given `Axis`.
///
/// The size of `Infer` is always `0`.
///
/// ## Examples
///
/// ```gleam
/// > size(A(1))
/// 1
///
/// > size(Axis(name: "Sparkle", size: 99))
/// 99
///
/// > size(Infer("Silver"))
/// 0
/// ```
pub fn size(x: Axis) -> Int {
  case x {
    Infer(_) -> 0
    Axis(size: size, ..) | A(size) | B(size) | C(size) | D(size) | E(size) | F(
      size,
    ) | G(size) | H(size) | I(size) | J(size) | K(size) | L(size) | M(size) | N(
      size,
    ) | O(size) | P(size) | Q(size) | R(size) | S(size) | T(size) | U(size) | V(
      size,
    ) | W(size) | X(size) | Y(size) | Z(size) -> size
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Transformation Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Renames the given `Axis`, retaining its `size`.
///
/// If an `Axis` is renamed to a single capital letter (from `"A"` to `"Z"`
/// inclusive), the single-letter convenience constructor will be used for the
/// new `Axis`. 
///
/// ## Examples
///
/// ```gleam
/// > let x = A(1)
/// > rename(x, "B")
/// B(1)
///
/// > let x = Axis("Thing", 3)
/// > rename(x, "Y")
/// Y(3)
///
/// > let x = Axis(name: "Sparkle", size: 99)
/// > rename(x, "Shine")
/// Axis("Shine", 99)
///
/// > let x = Infer("Silver")
/// > rename(x, "Gold")
/// Infer("Gold")
/// ```
///
pub fn rename(x: Axis, name: String) -> Axis {
  let size = size(x)
  let is_infer = case x {
    Infer(_) -> True
    _else -> False
  }
  case name {
    "A" -> A
    "B" -> B
    "C" -> C
    "D" -> D
    "E" -> E
    "F" -> F
    "G" -> G
    "H" -> H
    "I" -> I
    "J" -> J
    "K" -> K
    "L" -> L
    "M" -> M
    "N" -> N
    "O" -> O
    "P" -> P
    "Q" -> Q
    "R" -> R
    "S" -> S
    "T" -> T
    "U" -> U
    "V" -> V
    "W" -> W
    "X" -> X
    "Y" -> Y
    "Z" -> Z
    _else if is_infer -> fn(_) { Infer(name) }
    _else -> Axis(name: name, size: _)
  }(
    size,
  )
}

/// Changes the `size` of the given `Axis`.
///
/// Resizing an `Infer` returns an `Axis` record that will no longer have its
/// `size` automatically computed.
///
/// ## Examples
///
/// ```gleam
/// > let x = A(1)
/// > resize(x, 3)
/// A(3)
///
/// > let x = Axis("Y", 2)
/// > resize(x, 3)
/// Y(3)
///
/// > let x = Axis(name: "Sparkle", size: 99)
/// > resize(x, 42)
/// Axis("Sparkle", 42)
///
/// > let x = Infer("A")
/// > resize(x, 1)
/// A(1)
/// ```
///
pub fn resize(x: Axis, size: Int) -> Axis {
  case name(x) {
    "A" -> A
    "B" -> B
    "C" -> C
    "D" -> D
    "E" -> E
    "F" -> F
    "G" -> G
    "H" -> H
    "I" -> I
    "J" -> J
    "K" -> K
    "L" -> L
    "M" -> M
    "N" -> N
    "O" -> O
    "P" -> P
    "Q" -> Q
    "R" -> R
    "S" -> S
    "T" -> T
    "U" -> U
    "V" -> V
    "W" -> W
    "X" -> X
    "Y" -> Y
    "Z" -> Z
    name -> Axis(name: name, size: _)
  }(
    size,
  )
}
