import argamak/axis.{A, B, C, D, E, F, Infer, Z}
import argamak/format
import argamak/space
import argamak/tensor.{Tensor}
import gleam/dynamic.{Dynamic}
import gleam/float
import gleam/int
import gleam/list
import gleam/order.{Eq}
import gleeunit/should

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Creation Functions                     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn from_float_test() {
  0.0
  |> tensor.from_float
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))
}

pub fn from_int_test() {
  0
  |> tensor.from_int
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(0))
}

pub fn from_bool_test() {
  True
  |> tensor.from_bool
  |> should_share_native_format
  |> tensor.to_bool
  |> should.equal(Ok(True))

  False
  |> tensor.from_bool
  |> tensor.to_bool
  |> should.equal(Ok(False))
}

pub fn from_floats_test() {
  let xs =
    list.range(from: 1, to: 64)
    |> list.map(with: int.to_float)

  let d0 = space.new()
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(2), B(32))
  assert Ok(d3) = space.d3(A(2), B(2), C(16))
  assert Ok(d4) = space.d4(A(2), B(2), C(2), D(8))
  assert Ok(d5) = space.d5(A(2), B(2), C(2), D(2), E(4))
  assert Ok(d6) = space.d6(A(2), B(2), C(2), D(2), E(2), F(2))

  xs
  |> tensor.from_floats(into: d0)
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_floats(of: xs, into: d1)
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d2)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d3)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d4)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d5)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d6)
  x
  |> tensor.to_floats
  |> should.equal(xs)
}

pub fn from_ints_test() {
  let xs = list.range(from: 1, to: 64)

  let d0 = space.new()
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(2), B(32))
  assert Ok(d3) = space.d3(A(2), B(2), C(16))
  assert Ok(d4) = space.d4(A(2), B(2), C(2), D(8))
  assert Ok(d5) = space.d5(A(2), B(2), C(2), D(2), E(4))
  assert Ok(d6) = space.d6(A(2), B(2), C(2), D(2), E(2), F(2))

  xs
  |> tensor.from_ints(into: d0)
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: xs, into: d1)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d2)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d3)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d4)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d5)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d6)
  x
  |> tensor.to_ints
  |> should.equal(xs)
}

pub fn from_bools_test() {
  let xs =
    1
    |> list.range(to: 64)
    |> list.map(with: fn(x) {
      case x % 3 {
        0 -> False
        _else -> True
      }
    })

  let d0 = space.new()
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(2), B(32))
  assert Ok(d3) = space.d3(A(2), B(2), C(16))
  assert Ok(d4) = space.d4(A(2), B(2), C(2), D(8))
  assert Ok(d5) = space.d5(A(2), B(2), C(2), D(2), E(4))
  assert Ok(d6) = space.d6(A(2), B(2), C(2), D(2), E(2), F(2))

  xs
  |> tensor.from_bools(into: d0)
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_bools(of: xs, into: d1)
  x
  |> should_share_native_format
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_bools(of: xs, into: d2)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_bools(of: xs, into: d3)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_bools(of: xs, into: d4)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_bools(of: xs, into: d5)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_bools(of: xs, into: d6)
  x
  |> tensor.to_bools
  |> should.equal(xs)
}

pub fn from_native_test() {
  assert Ok(space) = space.d2(A(2), Infer("B"))

  assert Ok(x) =
    [[1, 2], [3, 4]]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: space, with: format.int32())
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 2, 3, 4])
  x
  |> tensor.axes
  |> should.equal([A(2), B(2)])
}

if erlang {
  external fn native_tensor(Dynamic) -> tensor.Native =
    "Elixir.Nx" "tensor"
}

if javascript {
  external fn native_tensor(Dynamic) -> tensor.Native =
    "../argamak_test_ffi.mjs" "tensor"
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reflection Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn format_test() {
  assert Ok(d1) = space.d1(Infer("A"))

  0.0
  |> tensor.from_float
  |> tensor.format
  |> should.equal(format.float32())

  0
  |> tensor.from_int
  |> tensor.format
  |> should.equal(format.int32())

  assert Ok(x) = tensor.from_floats([0.0], into: d1)
  x
  |> tensor.format
  |> should.equal(format.float32())

  assert Ok(x) = tensor.from_ints(of: [0], into: d1)
  x
  |> tensor.format
  |> should.equal(format.int32())
}

pub fn space_test() {
  let xs = [1, 2, 3, 4, 5, 6, 7, 8]

  0.0
  |> tensor.from_float
  |> tensor.space
  |> should.equal(space.new())

  assert Ok(space) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_ints(of: xs, into: space)
  x
  |> tensor.space
  |> space.axes
  |> should.equal([A(8)])

  assert Ok(space) = space.d3(A(2), B(2), C(2))
  assert Ok(x) = tensor.from_ints(of: xs, into: space)
  x
  |> tensor.space
  |> should.equal(space)
}

pub fn axes_test() {
  let xs = [0.0]

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(1), B(1))
  assert Ok(d3) = space.d3(A(1), B(1), C(1))
  assert Ok(d4) = space.d4(A(1), B(1), C(1), D(1))
  assert Ok(d5) = space.d5(A(1), B(1), C(1), D(1), E(1))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), E(1), F(1))
  assert Ok(d7) = space.from_list([A(1), B(1), C(1), D(1), E(1), F(1), Z(1)])

  0.0
  |> tensor.from_float
  |> tensor.axes
  |> should.equal([])

  assert Ok(x) = tensor.from_floats(of: xs, into: d1)
  x
  |> tensor.axes
  |> should.equal([A(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d2)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d3)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d4)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1), D(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d5)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1), D(1), E(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d6)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1), D(1), E(1), F(1)])

  assert Ok(x) = tensor.from_floats(of: xs, into: d7)
  x
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1), D(1), E(1), F(1), Z(1)])
}

pub fn rank_test() {
  let xs = [0.0]

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(1), B(1))
  assert Ok(d3) = space.d3(A(1), B(1), C(1))
  assert Ok(d4) = space.d4(A(1), B(1), C(1), D(1))
  assert Ok(d5) = space.d5(A(1), B(1), C(1), D(1), E(1))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), E(1), F(1))
  assert Ok(d7) = space.from_list([A(1), B(1), C(1), D(1), E(1), F(1), Z(1)])

  0.0
  |> tensor.from_float
  |> tensor.rank
  |> should.equal(0)

  assert Ok(x) = tensor.from_floats(of: xs, into: d1)
  x
  |> tensor.rank
  |> should.equal(1)

  assert Ok(x) = tensor.from_floats(of: xs, into: d2)
  x
  |> tensor.rank
  |> should.equal(2)

  assert Ok(x) = tensor.from_floats(of: xs, into: d3)
  x
  |> tensor.rank
  |> should.equal(3)

  assert Ok(x) = tensor.from_floats(of: xs, into: d4)
  x
  |> tensor.rank
  |> should.equal(4)

  assert Ok(x) = tensor.from_floats(of: xs, into: d5)
  x
  |> tensor.rank
  |> should.equal(5)

  assert Ok(x) = tensor.from_floats(of: xs, into: d6)
  x
  |> tensor.rank
  |> should.equal(6)

  assert Ok(x) = tensor.from_floats(of: xs, into: d7)
  x
  |> tensor.rank
  |> should.equal(7)
}

pub fn shape_test() {
  let xs =
    list.range(from: 1, to: 720)
    |> list.map(with: int.to_float)

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(1), B(2))
  assert Ok(d3) = space.d3(A(1), B(2), C(3))
  assert Ok(d4) = space.d4(A(1), B(2), C(3), D(4))
  assert Ok(d5) = space.d5(A(1), B(2), C(3), D(4), E(5))
  assert Ok(d6) = space.d6(A(1), B(2), C(3), D(4), E(5), F(6))
  assert Ok(d7) = space.from_list([A(1), B(2), C(3), D(4), E(5), F(6), Z(1)])

  0.0
  |> tensor.from_float
  |> tensor.shape
  |> should.equal([])

  assert Ok(x) = tensor.from_floats(of: [1.0], into: d1)
  x
  |> tensor.shape
  |> should.equal([1])

  assert Ok(x) = tensor.from_floats(of: [1.0, 2.0], into: d2)
  x
  |> tensor.shape
  |> should.equal([1, 2])

  assert Ok(x) =
    tensor.from_floats(of: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], into: d3)
  x
  |> tensor.shape
  |> should.equal([1, 2, 3])

  assert Ok(x) =
    tensor.from_floats(of: list.take(from: xs, up_to: 24), into: d4)
  x
  |> tensor.shape
  |> should.equal([1, 2, 3, 4])

  assert Ok(x) =
    tensor.from_floats(of: list.take(from: xs, up_to: 120), into: d5)
  x
  |> tensor.shape
  |> should.equal([1, 2, 3, 4, 5])

  assert Ok(x) = tensor.from_floats(of: xs, into: d6)
  x
  |> tensor.shape
  |> should.equal([1, 2, 3, 4, 5, 6])

  assert Ok(x) = tensor.from_floats(of: xs, into: d7)
  x
  |> tensor.shape
  |> should.equal([1, 2, 3, 4, 5, 6, 1])
}

pub fn size_test() {
  0.0
  |> tensor.from_float
  |> tensor.size
  |> should.equal(1)

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_ints(of: [1, 2, 3], into: d1)
  x
  |> tensor.size
  |> should.equal(3)

  assert Ok(d3) = space.d3(A(2), B(2), C(2))
  assert Ok(x) = tensor.from_ints(of: [1, 2, 3, 4, 5, 6, 7, 8], into: d3)
  x
  |> tensor.size
  |> should.equal(8)
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Transformation Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn reformat_test() {
  0
  |> tensor.from_int
  |> tensor.reformat(apply: format.float32())
  |> should_share_native_format
  |> tensor.format
  |> should.equal(format.float32())

  0.0
  |> tensor.from_float
  |> tensor.reformat(apply: format.int32())
  |> should_share_native_format
  |> tensor.format
  |> should.equal(format.int32())
}

pub fn broadcast_test() {
  assert Ok(d1) = space.d1(A(3))
  assert Ok(d2) = space.d2(A(2), B(3))

  assert Ok(x) =
    0
    |> tensor.from_int
    |> tensor.broadcast(into: d1)
  x
  |> should_share_native_format
  |> tensor.space
  |> space.axes
  |> should.equal(space.axes(d1))
  x
  |> tensor.to_ints
  |> should.equal([0, 0, 0])

  assert Ok(x) = tensor.broadcast(from: x, into: d2)
  x
  |> tensor.space
  |> space.axes
  |> should.equal(space.axes(d2))
  x
  |> tensor.to_ints
  |> should.equal([0, 0, 0, 0, 0, 0])
}

pub fn broadcast_over_test() {
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(A(3), B(2))
  assert Ok(d3) = space.d3(A(3), B(2), C(2))

  let xs = [1, 2, 3]
  assert Ok(x) = tensor.from_ints(of: xs, into: d1)
  assert Ok(x) = tensor.broadcast_over(from: x, into: d2, with: fn(_) { "A" })
  x
  |> should_share_native_format
  |> tensor.space
  |> space.axes
  |> should.equal(space.axes(d2))
  x
  |> tensor.to_ints
  |> should.equal(list.flat_map(over: xs, with: list.repeat(item: _, times: 2)))

  let xs = [1, 2, 3, 4, 5, 6]

  assert Ok(x) = tensor.from_ints(of: xs, into: d2)
  assert Ok(y) = tensor.broadcast_over(from: x, into: d3, with: axis.name)
  y
  |> tensor.space
  |> space.axes
  |> should.equal(space.axes(d3))
  y
  |> tensor.to_ints
  |> should.equal(list.flat_map(over: xs, with: list.repeat(item: _, times: 2)))

  assert Ok(y) =
    tensor.broadcast_over(
      from: x,
      into: d3,
      with: fn(axis) {
        case axis.name(axis) {
          "A" -> "A"
          "B" -> "C"
          name -> name
        }
      },
    )
  y
  |> tensor.space
  |> space.axes
  |> should.equal(space.axes(d3))
  y
  |> tensor.to_ints
  |> should.equal(
    xs
    |> list.sized_chunk(into: 2)
    |> list.flat_map(with: list.repeat(item: _, times: 2))
    |> list.flatten,
  )
}

pub fn reshape_test() {
  let d0 = space.new()
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(Infer("A"), B(1))
  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(d4) = space.d4(A(1), B(1), Infer("C"), D(1))
  assert Ok(d5) = space.d5(A(1), B(1), C(1), Infer("D"), E(1))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), Infer("E"), F(1))

  assert Ok(x) =
    0.0
    |> tensor.from_float
    |> tensor.reshape(into: d1)
  x
  |> should_share_native_format
  |> tensor.shape
  |> should.equal([1])

  assert Ok(x) = tensor.reshape(put: x, into: d2)
  x
  |> tensor.shape
  |> should.equal([1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d3)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d4)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d5)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d6)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d5)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d4)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d3)
  x
  |> tensor.shape
  |> should.equal([1, 1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d2)
  x
  |> tensor.shape
  |> should.equal([1, 1])

  assert Ok(x) = tensor.reshape(put: x, into: d1)
  x
  |> tensor.shape
  |> should.equal([1])

  assert Ok(x) = tensor.reshape(put: x, into: d0)
  x
  |> tensor.shape
  |> should.equal([])
}

pub fn squeeze_test() {
  0
  |> tensor.from_int
  |> tensor.squeeze(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.axes
  |> should.equal([])

  3.0
  |> tensor.from_float
  |> tensor.squeeze(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.axes
  |> should.equal([])

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.axes
  |> should.equal([])
  x
  |> tensor.squeeze(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [1, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.axes
  |> should.equal([A(2)])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.axes
  |> should.equal([B(2)])
  x
  |> tensor.squeeze(with: fn(x) { axis.name(x) == "C" })
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Logical Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn broadcastable_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.equal(is: a, to: tensor.from_int(4))
  x
  |> should_share_native_format
  |> tensor.axes
  |> should.equal([B(2)])

  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_ints(of: [4, 4, 5, 5], into: d2)
  assert Ok(x) = tensor.equal(is: a, to: b)
  x
  |> tensor.axes
  |> should.equal([A(2), B(2)])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(c) = tensor.from_ints(of: [4, 5, 6], into: d3)
  b
  |> tensor.equal(to: c)
  |> should.be_error

  assert Ok(d3) = space.d3(C(1), Infer("A"), B(1))
  assert Ok(c) = tensor.reshape(put: c, into: d3)
  b
  |> tensor.equal(to: c)
  |> should.be_error

  assert Ok(a) = tensor.from_floats(of: [5.0, 4.0], into: d1)
  assert Ok(x) = tensor.equal(is: a, to: tensor.from_float(4.0))
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([0.0, 1.0])

  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.equal(is: a, to: b)
  x
  |> tensor.to_floats
  |> should.equal([0.0, 1.0, 1.0, 0.0])
}

pub fn equal_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.equal(is: a, to: tensor.from_int(4))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.equal(is: a, to: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 1, 1, 0])
}

pub fn not_equal_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.not_equal(is: a, to: tensor.from_int(4))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.not_equal(is: a, to: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0, 0, 1])
}

pub fn greater_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.greater(is: a, than: tensor.from_int(4))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.greater(is: a, than: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0, 0, 0])
}

pub fn greater_or_equal_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.greater_or_equal(is: a, to: tensor.from_int(4))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.greater_or_equal(is: a, to: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1, 1, 0])
}

pub fn less_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.less(is: a, than: tensor.from_int(5))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.less(is: a, than: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 0, 0, 1])
}

pub fn less_or_equal_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [5, 4], into: d1)
  assert Ok(x) = tensor.less_or_equal(is: a, to: tensor.from_int(5))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [4.0, 4.0, 5.0, 5.0], into: d2)
  assert Ok(x) = tensor.less_or_equal(is: a, to: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 1, 1, 1])
}

pub fn logical_and_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [9, 0], into: d1)
  assert Ok(x) = tensor.logical_and(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 0.0], into: d2)
  assert Ok(x) = tensor.logical_and(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 0, 1, 0])
}

pub fn logical_or_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [9, 0], into: d1)
  assert Ok(x) = tensor.logical_or(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 0.0], into: d2)
  assert Ok(x) = tensor.logical_or(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1, 1, 0])
}

pub fn logical_xor_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [9, 0], into: d1)
  assert Ok(x) = tensor.logical_xor(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 1])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 0.0], into: d2)
  assert Ok(x) = tensor.logical_xor(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 1, 0, 0])
}

pub fn logical_not_test() {
  3
  |> tensor.from_int
  |> tensor.logical_not
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(0))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.3], into: d1)
  x
  |> tensor.logical_not
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([0.0])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints([-1, 8, 0], into: d3)
  let x = tensor.logical_not(x)
  x
  |> tensor.to_ints
  |> should.equal([0, 0, 1])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Arithmetic Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn add_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [0, 9], into: d1)
  assert Ok(x) = tensor.add(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([3, 12])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 0.0], into: d2)
  assert Ok(x) = tensor.add(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 13, 5, 9])
}

pub fn subtract_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [0, 9], into: d1)
  assert Ok(x) = tensor.subtract(from: a, value: tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([-3, 6])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 0.0], into: d2)
  assert Ok(x) = tensor.subtract(from: a, value: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 5, -5, 9])
}

pub fn multiply_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.multiply(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([3, 27])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 9.0], into: d2)
  assert Ok(x) = tensor.multiply(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 36, 5, 81])
}

pub fn divide_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.divide(from: a, by: tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 3])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, 9.0], into: d2)
  assert Ok(x) = tensor.divide(from: a, by: b)
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([0.0, 2.25, 0.2, 1.0])

  assert Ok(x) =
    [infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: space.new(), with: format.float32())
  x
  |> tensor.divide(by: x)
  |> should.equal(Error(tensor.InvalidData))
}

pub fn try_divide_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.try_divide(from: a, by: tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 3])

  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_ints(of: [0, 4, 5, 9], into: d2)
  a
  |> tensor.try_divide(by: b)
  |> should.equal(Error(tensor.ZeroDivision))
}

pub fn remainder_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [13, -13], into: d1)
  assert Ok(x) = tensor.remainder(from: a, divided_by: tensor.from_int(0))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 0])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [3.0, 3.0, -3.0, -3.0], into: d2)
  assert Ok(x) = tensor.remainder(from: a, divided_by: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, -1, 1, -1])
}

pub fn try_remainder_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.try_remainder(from: a, divided_by: tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 0])

  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_ints(of: [0, 4, 5, 9], into: d2)
  a
  |> tensor.try_remainder(divided_by: b)
  |> should.equal(Error(tensor.ZeroDivision))
}

pub fn modulo_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [13, -13], into: d1)
  assert Ok(x) = tensor.modulo(from: a, divided_by: tensor.from_int(0))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 0])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [3.0, 3.0, -3.0, -3.0], into: d2)
  assert Ok(x) = tensor.modulo(from: a, divided_by: b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 2, -2, -1])
}

pub fn try_modulo_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.try_modulo(from: a, divided_by: tensor.from_int(3))
  x
  |> tensor.to_ints
  |> should.equal([1, 0])

  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_ints(of: [0, 4, 5, 9], into: d2)
  a
  |> tensor.try_modulo(divided_by: b)
  |> should.equal(Error(tensor.ZeroDivision))
}

pub fn power_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.power(raise: a, to_the: tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 729])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 0.4, 0.5, 0.9], into: d2)
  assert Ok(x) = tensor.power(raise: a, to_the: b)
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([1.0, 2.408, 1.0, 7.225])
}

pub fn max_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.max(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([3, 9])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, -9.0], into: d2)
  assert Ok(x) = tensor.max(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 9, 5, 9])
}

pub fn min_test() {
  assert Ok(d1) = space.d1(Infer("B"))
  assert Ok(a) = tensor.from_ints(of: [1, 9], into: d1)
  assert Ok(x) = tensor.min(a, tensor.from_int(3))
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([1, 3])

  let a = tensor.reformat(a, apply: format.float32())
  assert Ok(d2) = space.d2(Infer("A"), B(2))
  assert Ok(b) = tensor.from_floats(of: [0.0, 4.0, 5.0, -9.0], into: d2)
  assert Ok(x) = tensor.min(a, b)
  x
  |> should_share_native_format
  |> tensor.to_ints
  |> should.equal([0, 4, 1, -9])
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Basic Math Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn absolute_value_test() {
  3
  |> tensor.from_int
  |> tensor.absolute_value
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.3], into: d1)
  x
  |> tensor.absolute_value
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([0.3])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints([-1, 8, 0], into: d3)
  let x = tensor.absolute_value(x)
  x
  |> tensor.to_ints
  |> should.equal([1, 8, 0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn negate_test() {
  3
  |> tensor.from_int
  |> tensor.negate
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(-3))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.3], into: d1)
  x
  |> tensor.negate
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([0.3])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints([-1, 8, 0], into: d3)
  let x = tensor.negate(x)
  x
  |> tensor.to_ints
  |> should.equal([1, -8, 0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn sign_test() {
  3
  |> tensor.from_int
  |> tensor.sign
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.3], into: d1)
  x
  |> tensor.sign
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([-1.0])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints([-1, 8, 0], into: d3)
  let x = tensor.sign(x)
  x
  |> tensor.to_ints
  |> should.equal([-1, 1, 0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn ceiling_test() {
  3
  |> tensor.from_int
  |> tensor.ceiling
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.5], into: d1)
  x
  |> tensor.ceiling
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([0.0])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([-1.2, 7.8, 0.0], into: d3)
  let x = tensor.ceiling(x)
  x
  |> tensor.to_floats
  |> should.equal([-1.0, 8.0, 0.0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn floor_test() {
  3
  |> tensor.from_int
  |> tensor.floor
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-0.5], into: d1)
  x
  |> tensor.floor
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([-1.0])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([-1.2, 7.8, 0.0], into: d3)
  let x = tensor.floor(x)
  x
  |> tensor.to_floats
  |> should.equal([-2.0, 7.0, 0.0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn round_test() {
  3
  |> tensor.from_int
  |> tensor.round
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  // For (+/-)0.5, TensorFlow currently rounds to 0.
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-1.5], into: d1)
  x
  |> tensor.round
  |> should_share_native_format
  |> tensor.to_floats
  |> should.equal([-2.0])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([-1.2, 7.8, 0.0], into: d3)
  let x = tensor.round(x)
  x
  |> tensor.to_floats
  |> should.equal([-1.0, 8.0, 0.0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])
}

pub fn exp_test() {
  3
  |> tensor.from_int
  |> tensor.exp
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(20))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([-1.5], into: d1)
  x
  |> tensor.exp
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([0.223])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([-1.2, 7.8, 0.0], into: d3)
  let x = tensor.exp(x)
  x
  |> tensor.to_floats
  |> should_loosely_equal([0.301, 2440.603, 1.0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])

  assert Ok(x) = tensor.from_ints([-90, 90, 0], into: d1)
  x
  |> tensor.exp
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(A(3)),
  [         0, 2147483647,          1],
)",
  )
}

pub fn square_root_test() {
  assert Ok(x) =
    3
    |> tensor.from_int
    |> tensor.square_root
  x
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([1.5], into: d1)
  assert Ok(x) = tensor.square_root(x)
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([1.225])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([1.2, 7.8, 0.0], into: d3)
  assert Ok(x) = tensor.square_root(x)
  x
  |> tensor.to_floats
  |> should_loosely_equal([1.095, 2.793, 0.0])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])

  assert Ok(x) = tensor.from_ints([1, 90, 0], into: d1)
  assert Ok(x) = tensor.square_root(x)
  x
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(A(3)),
  [1, 9, 0],
)",
  )

  -1
  |> tensor.from_int
  |> tensor.square_root
  |> should.equal(Error(tensor.InvalidData))

  assert Ok(x) = tensor.from_floats(of: [-0.1], into: d1)
  x
  |> tensor.square_root
  |> should.equal(Error(tensor.InvalidData))
}

pub fn ln_test() {
  assert Ok(x) =
    3
    |> tensor.from_int
    |> tensor.ln
  x
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(x) = tensor.from_floats([1.5], into: d1)
  assert Ok(x) = tensor.ln(x)
  x
  |> should_share_native_format
  |> tensor.to_floats
  |> should_loosely_equal([0.405])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_floats([1.2, 7.8, 0.0], into: d3)
  assert Ok(x) = tensor.ln(x)
  x
  |> tensor.to_floats
  |> should_loosely_equal([0.182, 2.054, float32_min])
  x
  |> tensor.axes
  |> should.equal([A(1), B(3), C(1)])

  assert Ok(x) = tensor.from_ints([1, 90, 0], into: d1)
  assert Ok(x) = tensor.ln(x)
  x
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(A(3)),
  [          0,           4, -2147483648],
)",
  )

  -1
  |> tensor.from_int
  |> tensor.ln
  |> should.equal(Error(tensor.InvalidData))

  assert Ok(x) = tensor.from_floats(of: [-0.1], into: d1)
  x
  |> tensor.ln
  |> should.equal(Error(tensor.InvalidData))
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Reduction Functions                    //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn in_situ_test() {
  0
  |> tensor.from_int
  |> tensor.in_situ_all(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(0))

  -3.0
  |> tensor.from_float
  |> tensor.in_situ_all(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(1.0))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  let y = tensor.in_situ_all(from: x, with: fn(_) { True })
  y
  |> tensor.to_ints
  |> should.equal([1])
  y
  |> tensor.axes
  |> should.equal([A(1)])
  let y = tensor.in_situ_all(from: x, with: fn(_) { False })
  y
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [0, 1]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.in_situ_all(with: fn(_) { True })
  |> tensor.axes
  |> should.equal([A(1)])

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  let y = tensor.in_situ_all(from: x, with: fn(_) { True })
  y
  |> tensor.axes
  |> should.equal([A(1), B(1), C(1)])
  y
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))
  let y = tensor.in_situ_all(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.axes
  |> should.equal([A(1), B(2), C(1)])
  y
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.to_ints
  |> should.equal([0, 1])
}

pub fn all_test() {
  0.0
  |> tensor.from_float
  |> tensor.all(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.all(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.all(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))
  x
  |> tensor.all(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [0, 1]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.all(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.all(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))
  let y = tensor.all(from: x, with: fn(x) { axis.name(x) == "B" })
  y
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))
  y
  |> tensor.axes
  |> should.equal([A(1), C(1)])
}

pub fn any_test() {
  0.0
  |> tensor.from_float
  |> tensor.any(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.any(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.any(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))
  x
  |> tensor.any(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [0, 1]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.any(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(1))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.any(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))
  let y = tensor.any(from: x, with: fn(x) { axis.name(x) == "B" })
  y
  |> tensor.squeeze(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))
  y
  |> tensor.axes
  |> should.equal([A(1), C(1)])
}

pub fn arg_max_test() {
  0.0
  |> tensor.from_float
  |> tensor.arg_max(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.arg_max(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(0))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.arg_max(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))
  x
  |> tensor.arg_max(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([])

  let xs = [1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.arg_max(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.arg_max(with: fn(_) { True })
  |> tensor.to_ints
  |> should.equal([0, 0, 0, 0])
  let y = tensor.arg_max(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([1, 0])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn arg_min_test() {
  0.0
  |> tensor.from_float
  |> tensor.arg_min(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.arg_min(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(0))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.arg_min(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))
  x
  |> tensor.arg_min(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([])

  let xs = [1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.arg_min(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(0))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.arg_min(with: fn(_) { True })
  |> tensor.to_ints
  |> should.equal([0, 0, 0, 0])
  let y = tensor.arg_min(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([0, 1])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn max_over_test() {
  0.0
  |> tensor.from_float
  |> tensor.max_over(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.max_over(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.max_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(3))
  x
  |> tensor.max_over(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.max_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(4))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.max_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(4))
  let y = tensor.max_over(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([4, 3])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn min_over_test() {
  0.0
  |> tensor.from_float
  |> tensor.min_over(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.min_over(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.min_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(3))
  x
  |> tensor.min_over(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.min_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.min_over(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(1))
  let y = tensor.min_over(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([1, 2])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn sum_test() {
  0.0
  |> tensor.from_float
  |> tensor.sum(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.sum(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.sum(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(3))
  x
  |> tensor.sum(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [-1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.sum(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(8))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.sum(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(8))
  let y = tensor.sum(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([3, 5])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn product_test() {
  0.0
  |> tensor.from_float
  |> tensor.product(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.product(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.product(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(3))
  x
  |> tensor.product(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [-1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.product(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(-24))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  x
  |> tensor.product(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(-24))
  let y = tensor.product(from: x, with: fn(x) { axis.name(x) == "C" })
  y
  |> tensor.to_ints
  |> should.equal([-4, 6])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])
}

pub fn mean_test() {
  0.0
  |> tensor.from_float
  |> tensor.mean(with: fn(_) { True })
  |> should_share_native_format
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  3
  |> tensor.from_int
  |> tensor.mean(with: fn(_) { False })
  |> should_share_native_format
  |> tensor.to_int
  |> should.equal(Ok(3))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_ints(of: [3], into: d1)
  x
  |> tensor.mean(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(3))
  x
  |> tensor.mean(with: fn(_) { False })
  |> tensor.axes
  |> should.equal([A(1)])

  let xs = [-1, 4, 3, 2]

  assert Ok(x) = tensor.from_ints(xs, into: d1)
  x
  |> tensor.mean(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(2))

  assert Ok(d3) = space.d3(A(1), Infer("B"), C(2))
  assert Ok(x) = tensor.from_ints(xs, into: d3)
  let f = fn(x) { axis.name(x) == "C" }

  x
  |> tensor.mean(with: fn(_) { True })
  |> tensor.to_int
  |> should.equal(Ok(2))
  let y = tensor.mean(from: x, with: f)
  y
  |> tensor.to_ints
  |> should.equal([1, 2])
  y
  |> tensor.axes
  |> should.equal([A(1), B(2)])

  x
  |> tensor.reformat(apply: format.float32())
  |> tensor.mean(with: f)
  |> tensor.to_floats
  |> should_loosely_equal([1.5, 2.5])
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Conversion Functions                   //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

const float32_min = -340_282_346_638_528_859_811_704_183_484_516_925_440.0

const float32_max = 340_282_346_638_528_859_811_704_183_484_516_925_440.0

const int32_min = -2_147_483_648

const int32_max = 2_147_483_647

pub fn to_float_test() {
  0.0
  |> tensor.from_float
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  0
  |> tensor.from_int
  |> tensor.to_float
  |> should.equal(Ok(0.0))

  let d0 = space.new()

  assert Ok(x) =
    [neg_infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_float
  |> should.equal(Ok(float32_min))

  assert Ok(x) =
    [infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_float
  |> should.equal(Ok(float32_max))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_floats(of: [0.0], into: d1)
  x
  |> tensor.to_float
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_floats(of: [0.0, 1.0], into: d1)
  x
  |> tensor.to_float
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0], into: d1)
  x
  |> tensor.to_float
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0, 1], into: d1)
  x
  |> tensor.to_float
  |> should.equal(Error(tensor.IncompatibleShape))
}

pub fn to_int_test() {
  0.0
  |> tensor.from_float
  |> tensor.to_int
  |> should.equal(Ok(0))

  0
  |> tensor.from_int
  |> tensor.to_int
  |> should.equal(Ok(0))

  let d0 = space.new()

  assert Ok(x) =
    [neg_infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_int
  |> should.equal(Ok(int32_min))

  assert Ok(x) =
    [infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_int
  |> should.equal(Ok(int32_max))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_floats(of: [0.0], into: d1)
  x
  |> tensor.to_int
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_floats(of: [0.0, 1.0], into: d1)
  x
  |> tensor.to_int
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0], into: d1)
  x
  |> tensor.to_int
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0, 1], into: d1)
  x
  |> tensor.to_int
  |> should.equal(Error(tensor.IncompatibleShape))
}

pub fn to_bool_test() {
  0.0
  |> tensor.from_float
  |> tensor.to_bool
  |> should.equal(Ok(False))

  1
  |> tensor.from_int
  |> tensor.to_bool
  |> should.equal(Ok(True))

  let d0 = space.new()

  assert Ok(x) =
    [neg_infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_bool
  |> should.equal(Ok(True))

  assert Ok(x) =
    [infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d0, with: format.float32())
  x
  |> tensor.to_bool
  |> should.equal(Ok(True))

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_floats(of: [0.0], into: d1)
  x
  |> tensor.to_bool
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_floats(of: [0.0, 1.0], into: d1)
  x
  |> tensor.to_bool
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0], into: d1)
  x
  |> tensor.to_bool
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(x) = tensor.from_ints(of: [0, 1], into: d1)
  x
  |> tensor.to_bool
  |> should.equal(Error(tensor.IncompatibleShape))
}

pub fn to_floats_test() {
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), E(1), Infer("F"))

  let xs = [1.0, 2.0, 3.0]

  0.0
  |> tensor.from_float
  |> tensor.to_floats
  |> should.equal([0.0])

  assert Ok(x) = tensor.from_floats(of: xs, into: d1)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: xs, into: d6)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  let ys = [1, 2, 3]

  0
  |> tensor.from_int
  |> tensor.to_floats
  |> should.equal([0.0])

  assert Ok(x) = tensor.from_ints(of: ys, into: d1)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: ys, into: d6)
  x
  |> tensor.to_floats
  |> should.equal(xs)

  assert Ok(x) =
    [neg_infinity(), infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d1, with: format.float32())
  x
  |> tensor.to_floats
  |> should.equal([float32_min, float32_max])
}

pub fn to_ints_test() {
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), E(1), Infer("F"))

  let xs = [1, 2, 3]

  0
  |> tensor.from_int
  |> tensor.to_ints
  |> should.equal([0])

  assert Ok(x) = tensor.from_ints(of: xs, into: d1)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: xs, into: d6)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  let ys = [1.0, 2.0, 3.0]

  0.0
  |> tensor.from_float
  |> tensor.to_ints
  |> should.equal([0])

  assert Ok(x) = tensor.from_floats(of: ys, into: d1)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: ys, into: d6)
  x
  |> tensor.to_ints
  |> should.equal(xs)

  assert Ok(x) =
    [neg_infinity(), infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d1, with: format.float32())
  x
  |> tensor.to_ints
  |> should.equal([int32_min, int32_max])
}

pub fn to_bools_test() {
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d6) = space.d6(A(1), B(1), C(1), D(1), E(1), Infer("F"))

  let xs = [True, False, True]

  let ys = [1, 0, -3]

  1
  |> tensor.from_int
  |> tensor.to_bools
  |> should.equal([True])

  assert Ok(x) = tensor.from_ints(of: ys, into: d1)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_ints(of: ys, into: d6)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  let ys = [1.0, 0.0, -3.0]

  0.0
  |> tensor.from_float
  |> tensor.to_bools
  |> should.equal([False])

  assert Ok(x) = tensor.from_floats(of: ys, into: d1)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) = tensor.from_floats(of: ys, into: d6)
  x
  |> tensor.to_bools
  |> should.equal(xs)

  assert Ok(x) =
    [neg_infinity(), infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d1, with: format.float32())
  x
  |> tensor.to_bools
  |> should.equal([True, True])
}

pub fn to_native_test() {
  assert Ok(space) = space.d3(A(2), Infer("B"), C(2))
  assert Ok(x) =
    [1, 2, 3, 4, 5, 6, 7, 8]
    |> tensor.from_ints(into: space)
  x
  |> tensor.to_native
  |> native_shape
  |> should.equal(dynamic.from(#(2, 2, 2)))
}

if erlang {
  external fn native_shape(tensor.Native) -> Dynamic =
    "Elixir.Nx" "shape"
}

if javascript {
  external fn native_shape(tensor.Native) -> Dynamic =
    "../argamak_test_ffi.mjs" "shape"
}

pub fn to_string_test() {
  0.0
  |> tensor.from_float
  |> tensor.to_string(return: tensor.Data, wrap_at: 0)
  |> should.equal("0.0")

  0
  |> tensor.from_int
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(),
  0,
)",
  )

  assert Ok(d1) = space.d1(Infer("A"))

  assert Ok(x) = tensor.from_floats(of: [0.0], into: d1)
  x
  |> tensor.to_string(return: tensor.Data, wrap_at: 0)
  |> should.equal("[0.0]")

  assert Ok(x) = tensor.from_ints(of: [0], into: d1)
  x
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(A(1)),
  [0],
)",
  )

  assert Ok(x) = tensor.from_ints(of: [101, 3, 225, 4_000_000], into: d1)
  x
  |> tensor.to_string(return: tensor.Data, wrap_at: 30)
  |> should.equal("[    101,       3,     225,\n 4000000]")

  assert Ok(d2) = space.d2(A(2), B(2))
  assert Ok(x) =
    [0.0, 2.25, 0.20000000298023224, -1.0]
    |> tensor.from_floats(into: d2)
  x
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Float32),
  Space(A(2), B(2)),
  [[ 0.0, 2.25],
   [ 0.2, -1.0]],
)",
  )

  assert Ok(d4) = space.d4(A(1), Infer("B"), C(2), D(2))
  assert Ok(x) =
    [1, 2, 3, 44, 5, 6789, 10, 11, 12, 132, 5, 7]
    |> tensor.from_ints(into: d4)
  x
  |> tensor.to_string(return: tensor.Record, wrap_at: 0)
  |> should.equal(
    "Tensor(
  Format(Int32),
  Space(A(1), B(3), C(2), D(2)),
  [[[[   1,    2],
     [   3,   44]],
    [[   5, 6789],
     [  10,   11]],
    [[  12,  132],
     [   5,    7]]]],
)",
  )

  assert Ok(d6) = space.d6(A(1), B(1), Infer("C"), D(3), E(3), F(3))
  assert Ok(x) =
    1
    |> list.range(to: 54)
    |> tensor.from_ints(into: d6)
  x
  |> tensor.to_string(return: tensor.Data, wrap_at: 0)
  |> should.equal(
    "[[[[[[ 1,  2,  3],
     [ 4,  5,  6],
     [ 7,  8,  9]],
    [[10, 11, 12],
     [13, 14, 15],
     [16, 17, 18]],
    [[19, 20, 21],
     [22, 23, 24],
     [25, 26, 27]]],
   [[[28, 29, 30],
     [31, 32, 33],
     [34, 35, 36]],
    [[37, 38, 39],
     [40, 41, 42],
     [43, 44, 45]],
    [[46, 47, 48],
     [49, 50, 51],
     [52, 53, 54]]]]]]",
  )

  assert Ok(x) =
    [neg_infinity(), infinity()]
    |> dynamic.from
    |> native_tensor
    |> tensor.from_native(into: d1, with: format.float32())
  x
  |> tensor.to_string(return: tensor.Data, wrap_at: 0)
  |> should.equal("[-Infinity,  Infinity]")
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Private Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

pub fn fit_test() {
  assert Ok(d1) = space.d1(Infer("A"))
  assert Ok(d2) = space.d2(Infer("A"), B(1))

  0.0
  |> tensor.from_float
  |> tensor.shape
  |> should.equal([])

  assert Ok(x) = tensor.from_floats(of: [1.0, 2.0, 3.0], into: d1)
  x
  |> tensor.shape
  |> should.equal([3])

  assert Ok(x) = tensor.reshape(put: x, into: d2)
  x
  |> tensor.shape
  |> should.equal([3, 1])
}

fn should_share_native_format(x: Tensor(a)) -> Tensor(a) {
  x
  |> tensor.to_native
  |> native_format
  |> should.equal(format.to_native(tensor.format(x)))

  x
}

if erlang {
  external fn native_format(tensor.Native) -> format.Native =
    "Elixir.Nx" "type"
}

if javascript {
  external fn native_format(tensor.Native) -> format.Native =
    "../argamak_test_ffi.mjs" "type"
}

fn should_loosely_equal(a: List(Float), b: List(Float)) -> Nil {
  a
  |> list.zip(b)
  |> list.map(with: fn(pair) {
    float.loosely_compare(pair.0, with: pair.1, tolerating: 0.002)
  })
  |> list.all(satisfying: fn(x) { x == Eq })
  |> should.be_true
}

if erlang {
  type Infinity {
    Infinity
    NegInfinity
  }

  fn infinity() -> Dynamic {
    dynamic.from(Infinity)
  }

  fn neg_infinity() -> Dynamic {
    dynamic.from(NegInfinity)
  }
}

if javascript {
  external fn infinity() -> Dynamic =
    "../argamak_test_ffi.mjs" "infinity"

  external fn neg_infinity() -> Dynamic =
    "../argamak_test_ffi.mjs" "neg_infinity"
}
