import argamak/axis.{A, Axis, B, C, D, E, Infer, Z}
import argamak/space.{
  CannotInfer, CannotMerge, DuplicateName, InvalidSize, SpaceError,
}
import gleam/list
import gleeunit/should

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Creation Functions                     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn new_test() {
  space.new()
  |> space.axes
  |> should.equal([])
}

pub fn d1_test() {
  let a = A(size: 1)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  a
  |> space.d1
  |> should.be_ok

  axis
  |> space.d1
  |> should.be_ok

  infer
  |> space.d1
  |> should.be_ok

  let a = A(size: 0)
  space.d1(a)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d1(a)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))
}

pub fn d2_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  space.d2(a, axis)
  |> should.be_ok

  space.d2(a, infer)
  |> should.be_ok

  let a = A(size: 0)
  space.d2(a, b)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d2(a, b)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  space.d2(a, axis_a)
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  space.d2(Infer(name: "A"), infer)
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

pub fn d3_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  space.d3(a, b, axis)
  |> should.be_ok

  space.d3(a, infer, axis)
  |> should.be_ok

  let a = A(size: 0)
  space.d3(a, b, axis)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d3(a, b, infer)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  space.d3(a, axis_a, axis)
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  space.d3(Infer(name: "A"), infer, axis)
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

pub fn d4_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let c = C(size: 9)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  space.d4(a, b, c, axis)
  |> should.be_ok

  space.d4(a, b, infer, axis)
  |> should.be_ok

  let a = A(size: 0)
  space.d4(a, b, c, axis)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d4(a, b, c, infer)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  space.d4(a, b, axis_a, axis)
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  space.d4(Infer(name: "A"), infer, b, axis)
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

pub fn d5_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let c = C(size: 9)
  let d = D(size: 27)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  space.d5(a, b, c, d, axis)
  |> should.be_ok

  space.d5(a, b, c, infer, axis)
  |> should.be_ok

  let a = A(size: 0)
  space.d5(a, b, c, d, axis)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d5(a, b, c, d, infer)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  space.d5(a, b, c, axis_a, axis)
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  space.d5(Infer(name: "A"), infer, b, c, axis)
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

pub fn d6_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let c = C(size: 9)
  let d = D(size: 27)
  let e = E(size: 81)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  space.d6(a, b, c, d, e, axis)
  |> should.be_ok

  space.d6(a, b, c, d, infer, axis)
  |> should.be_ok

  let a = A(size: 0)
  space.d6(a, b, c, d, e, axis)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  space.d6(a, b, c, d, e, infer)
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  space.d6(a, b, c, d, axis_a, axis)
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  space.d6(Infer(name: "A"), infer, b, c, d, axis)
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

pub fn from_list_test() {
  let a = A(size: 1)
  let b = B(size: 3)
  let c = C(size: 9)
  let d = D(size: 27)
  let e = E(size: 81)
  let z = Z(size: 243)
  let axis = Axis(name: "Sparkle", size: 9)
  let infer = Infer(name: "Shine")

  [a, b, c, d, e, z, axis]
  |> space.from_list
  |> should.be_ok

  [a, b, c, d, e, infer, axis]
  |> space.from_list
  |> should.be_ok

  let a = A(size: 0)
  [a, b, c, d, e, z, axis]
  |> space.from_list
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let a = A(size: -1)
  [a, b, c, d, e, z, infer]
  |> space.from_list
  |> should.equal(Error([SpaceError(InvalidSize, [a])]))

  let axis_a = Axis(name: "A", size: 1)
  [a, b, c, d, e, axis_a, axis]
  |> space.from_list
  |> should.equal(Error([
    SpaceError(InvalidSize, [a]),
    SpaceError(DuplicateName, [axis_a]),
  ]))

  [Infer(name: "A"), infer, b, c, d, e, axis]
  |> space.from_list
  |> should.equal(Error([SpaceError(CannotInfer, [infer])]))
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reflection Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn axes_test() {
  space.new()
  |> space.axes
  |> should.equal([])

  let a = A(size: 1)

  assert Ok(d1) = space.d1(a)
  d1
  |> space.axes
  |> should.equal([a])

  let xs = [
    a,
    B(size: 3),
    C(size: 9),
    D(size: 27),
    E(size: 81),
    Z(size: 243),
    Infer(name: "Shine"),
    Axis(name: "Sparkle", size: 9),
  ]
  assert Ok(d8) = space.from_list(xs)
  d8
  |> space.axes
  |> should.equal(xs)
}

pub fn degree_test() {
  space.new()
  |> space.degree
  |> should.equal(0)

  let a = A(size: 1)

  assert Ok(d1) = space.d1(a)
  d1
  |> space.degree
  |> should.equal(1)

  let xs = [
    a,
    B(size: 3),
    C(size: 9),
    D(size: 27),
    E(size: 81),
    Z(size: 243),
    Infer(name: "Shine"),
    Axis(name: "Sparkle", size: 9),
  ]
  assert Ok(d8) = space.from_list(xs)
  d8
  |> space.degree
  |> should.equal(8)
}

pub fn shape_test() {
  space.new()
  |> space.shape
  |> should.equal([])

  let a = A(size: 1)

  assert Ok(d1) = space.d1(a)
  d1
  |> space.shape
  |> should.equal([1])

  let xs = [
    a,
    B(size: 3),
    C(size: 9),
    D(size: 27),
    E(size: 81),
    Z(size: 243),
    Infer(name: "Shine"),
    Axis(name: "Sparkle", size: 9),
  ]
  assert Ok(d8) = space.from_list(xs)
  d8
  |> space.shape
  |> should.equal([1, 3, 9, 27, 81, 243, 0, 9])
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Transformation Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn map_test() {
  let resize = axis.resize(_, 3)

  assert Ok(d0) =
    space.new()
    |> space.map(with: resize)
  d0
  |> space.axes
  |> should.equal([])

  let a = A(size: 1)
  assert Ok(d1) = space.d1(a)
  assert Ok(d1) = space.map(d1, with: resize)
  d1
  |> space.axes
  |> should.equal([A(size: 3)])

  let xs = [
    a,
    B(size: 3),
    C(size: 9),
    D(size: 27),
    E(size: 81),
    Z(size: 243),
    Infer(name: "Shine"),
    Axis(name: "Sparkle", size: 9),
  ]
  assert Ok(d8) = space.from_list(xs)
  assert Ok(d8) = space.map(d8, with: resize)

  d8
  |> space.axes
  |> list.map(with: axis.size)
  |> list.all(fn(x) { x == 3 })
  |> should.be_true

  d8
  |> space.map(with: axis.resize(_, 0))
  |> should.be_error

  d8
  |> space.map(with: axis.rename(_, "A"))
  |> should.be_error

  d8
  |> space.map(with: fn(a) {
    a
    |> axis.name
    |> Infer
  })
  |> should.be_error
}

pub fn merge_test() {
  let a = A(size: 1)
  let infer = Infer(name: "Shine")
  let axis = Axis(name: "Sparkle", size: 9)

  let d0 = space.new()

  assert Ok(d1) = space.d1(a)
  assert Ok(d1) = space.merge(d1, d0)
  d1
  |> space.axes
  |> should.equal([a])

  assert Ok(d2) = space.d2(a, axis)
  assert Ok(d2) = space.merge(d2, d0)
  d2
  |> space.axes
  |> should.equal([a, axis])

  assert Ok(d3) = space.d3(a, infer, axis)
  assert Ok(d3) = space.merge(d3, d0)
  d3
  |> space.axes
  |> should.equal([a, infer, axis])

  assert Ok(d1_axis) = space.d1(Axis(name: "A", size: 9))
  assert Ok(d1) = space.merge(d1, d1_axis)
  d1
  |> space.axes
  |> should.equal([A(size: 9)])

  assert Ok(d1) =
    axis
    |> axis.resize(1)
    |> space.d1
  assert Ok(d3) = space.merge(d3, d1)
  d3
  |> space.axes
  |> should.equal([a, infer, axis])

  assert Ok(d1) =
    axis
    |> axis.name
    |> Infer
    |> space.d1
  assert Ok(d3) = space.map(d3, with: axis.resize(_, 1))
  assert Ok(d3) = space.merge(d1, d3)
  d3
  |> space.axes
  |> should.equal([a, Axis(name: "Shine", size: 1), Infer(name: "Sparkle")])

  assert Ok(d1) = space.d1(a)
  space.merge(d1, d3)
  |> should.equal(Error([
    SpaceError(CannotMerge, [Infer(name: "Sparkle"), A(size: 1)]),
  ]))

  assert Ok(d2) = space.d2(Infer(name: "Shine"), axis)
  space.merge(d3, d2)
  |> should.equal(Error([SpaceError(CannotInfer, [Infer(name: "Sparkle")])]))
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Conversion Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn to_string_test() {
  space.new()
  |> space.to_string
  |> should.equal("Space()")

  let a = A(size: 1)

  assert Ok(d1) = space.d1(a)
  d1
  |> space.to_string
  |> should.equal("Space(A(1))")

  let xs = [
    a,
    B(size: 3),
    C(size: 9),
    D(size: 27),
    E(size: 81),
    Z(size: 243),
    Infer(name: "Shine"),
    Axis(name: "Sparkle", size: 9),
  ]
  assert Ok(d8) = space.from_list(xs)
  d8
  |> space.to_string
  |> should.equal(
    "Space(A(1), B(3), C(9), D(27), E(81), Z(243), Infer(\"Shine\"), Axis(\"Sparkle\", 9))",
  )
}
