import argamak/format
import argamak/space
import argamak/tensor
import gleam/dynamic.{Dynamic}
import gleam/int
import gleam/list
import gleam/result
import gleeunit/should

pub type Axis {
  A
  B
  C
  D
  E
  F
}

external fn erlang_tensor(Dynamic) -> tensor.Native =
  "Elixir.Nx" "tensor"

external fn erlang_shape(tensor.Native) -> Dynamic =
  "Elixir.Nx" "shape"

pub fn from_float_test() {
  0.
  |> tensor.from_float
  |> tensor.to_float
  |> should.equal(0.)
}

pub fn from_int_test() {
  0
  |> tensor.from_int
  |> tensor.to_int
  |> should.equal(0)
}

pub fn from_floats_test() {
  let list =
    list.range(from: 1, to: 65)
    |> list.map(with: int.to_float)

  assert Ok(d0) = space.d0()
  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, 2), #(B, 32))
  assert Ok(d3) = space.d3(#(A, 2), #(B, 2), #(C, 16))
  assert Ok(d4) = space.d4(#(A, 2), #(B, 2), #(C, 2), #(D, 8))
  assert Ok(d5) = space.d5(#(A, 2), #(B, 2), #(C, 2), #(D, 2), #(E, 4))
  assert Ok(d6) = space.d6(#(A, 2), #(B, 2), #(C, 2), #(D, 2), #(E, 2), #(F, 2))

  list
  |> tensor.from_floats(into: d0)
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(tensor) = tensor.from_floats(of: list, into: d1)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d2)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d3)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d4)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d5)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
  tensor
  |> tensor.to_list
  |> should.equal(list)
}

pub fn from_ints_test() {
  let list = list.range(from: 1, to: 65)

  assert Ok(d0) = space.d0()
  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, 2), #(B, 32))
  assert Ok(d3) = space.d3(#(A, 2), #(B, 2), #(C, 16))
  assert Ok(d4) = space.d4(#(A, 2), #(B, 2), #(C, 2), #(D, 8))
  assert Ok(d5) = space.d5(#(A, 2), #(B, 2), #(C, 2), #(D, 2), #(E, 4))
  assert Ok(d6) = space.d6(#(A, 2), #(B, 2), #(C, 2), #(D, 2), #(E, 2), #(F, 2))

  list
  |> tensor.from_ints(into: d0)
  |> should.equal(Error(tensor.IncompatibleShape))

  assert Ok(tensor) = tensor.from_ints(of: list, into: d1)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_ints(of: list, into: d2)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_ints(of: list, into: d3)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_ints(of: list, into: d4)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_ints(of: list, into: d5)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_ints(of: list, into: d6)
  tensor
  |> tensor.to_list
  |> should.equal(list)
}

pub fn from_native_test() {
  assert Ok(space) = space.d2(#(A, 2), #(B, -1))

  [[1, 2], [3, 4]]
  |> dynamic.from
  |> erlang_tensor
  |> tensor.from_native(into: space, with: format.int32)
  |> should.be_ok
}

pub fn axes_test() {
  let list = [0.]

  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, 1), #(B, 1))
  assert Ok(d3) = space.d3(#(A, 1), #(B, 1), #(C, 1))
  assert Ok(d4) = space.d4(#(A, 1), #(B, 1), #(C, 1), #(D, 1))
  assert Ok(d5) = space.d5(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1))
  assert Ok(d6) = space.d6(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1), #(F, 1))

  0.
  |> tensor.from_float
  |> tensor.axes
  |> should.equal([])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d1)
  tensor
  |> tensor.axes
  |> should.equal([A])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d2)
  tensor
  |> tensor.axes
  |> should.equal([A, B])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d3)
  tensor
  |> tensor.axes
  |> should.equal([A, B, C])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d4)
  tensor
  |> tensor.axes
  |> should.equal([A, B, C, D])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d5)
  tensor
  |> tensor.axes
  |> should.equal([A, B, C, D, E])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
  tensor
  |> tensor.axes
  |> should.equal([A, B, C, D, E, F])
}

pub fn format_test() {
  assert Ok(d1) = space.d1(A)

  0.
  |> tensor.from_float
  |> tensor.format
  |> should.equal(format.float32())

  0
  |> tensor.from_int
  |> tensor.format
  |> should.equal(format.int32())

  assert Ok(tensor) = tensor.from_floats([0.], into: d1)
  tensor
  |> tensor.format
  |> should.equal(format.float32())

  assert Ok(tensor) = tensor.from_ints([0], into: d1)
  tensor
  |> tensor.format
  |> should.equal(format.int32())
}

pub fn rank_test() {
  let list = [0.]

  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, 1), #(B, 1))
  assert Ok(d3) = space.d3(#(A, 1), #(B, 1), #(C, 1))
  assert Ok(d4) = space.d4(#(A, 1), #(B, 1), #(C, 1), #(D, 1))
  assert Ok(d5) = space.d5(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1))
  assert Ok(d6) = space.d6(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1), #(F, 1))

  0.
  |> tensor.from_float
  |> tensor.rank
  |> should.equal(0)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d1)
  tensor
  |> tensor.rank
  |> should.equal(1)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d2)
  tensor
  |> tensor.rank
  |> should.equal(2)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d3)
  tensor
  |> tensor.rank
  |> should.equal(3)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d4)
  tensor
  |> tensor.rank
  |> should.equal(4)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d5)
  tensor
  |> tensor.rank
  |> should.equal(5)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
  tensor
  |> tensor.rank
  |> should.equal(6)
}

pub fn shape_test() {
  let list =
    list.range(from: 1, to: 721)
    |> list.map(with: int.to_float)

  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, 1), #(B, 2))
  assert Ok(d3) = space.d3(#(A, 1), #(B, 2), #(C, 3))
  assert Ok(d4) = space.d4(#(A, 1), #(B, 2), #(C, 3), #(D, 4))
  assert Ok(d5) = space.d5(#(A, 1), #(B, 2), #(C, 3), #(D, 4), #(E, 5))
  assert Ok(d6) = space.d6(#(A, 1), #(B, 2), #(C, 3), #(D, 4), #(E, 5), #(F, 6))

  0.
  |> tensor.from_float
  |> tensor.shape
  |> should.equal([])

  assert Ok(tensor) = tensor.from_floats(of: [1.], into: d1)
  tensor
  |> tensor.shape
  |> should.equal([1])

  assert Ok(tensor) = tensor.from_floats(of: [1., 2.], into: d2)
  tensor
  |> tensor.shape
  |> should.equal([1, 2])

  assert Ok(tensor) = tensor.from_floats(of: [1., 2., 3., 4., 5., 6.], into: d3)
  tensor
  |> tensor.shape
  |> should.equal([1, 2, 3])

  assert Ok(tensor) =
    tensor.from_floats(of: list.take(from: list, up_to: 24), into: d4)
  tensor
  |> tensor.shape
  |> should.equal([1, 2, 3, 4])

  assert Ok(tensor) =
    tensor.from_floats(of: list.take(from: list, up_to: 120), into: d5)
  tensor
  |> tensor.shape
  |> should.equal([1, 2, 3, 4, 5])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
  tensor
  |> tensor.shape
  |> should.equal([1, 2, 3, 4, 5, 6])
}

pub fn space_test() {
  let list = [1, 2, 3, 4, 5, 6, 7, 8]

  assert Ok(space) = space.d0()
  0.
  |> tensor.from_float
  |> tensor.space
  |> should.equal(space)

  assert Ok(space) = space.d1(A)
  assert Ok(tensor) = tensor.from_ints(of: list, into: space)
  tensor
  |> tensor.space
  |> space.elements
  |> should.equal([#(A, 8)])

  assert Ok(space) = space.d3(#(A, 2), #(B, 2), #(C, 2))
  assert Ok(tensor) = tensor.from_ints(of: list, into: space)
  tensor
  |> tensor.space
  |> should.equal(space)
}

pub fn as_format_test() {
  0
  |> tensor.from_int
  |> tensor.as_format(apply: format.float32)
  |> tensor.format
  |> should.equal(format.float32())

  0.
  |> tensor.from_float
  |> tensor.as_format(apply: format.int32)
  |> tensor.format
  |> should.equal(format.int32())
}

pub fn fit_test() {
  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, -1), #(B, 1))

  0.
  |> tensor.from_float
  |> tensor.shape
  |> should.equal([])

  assert Ok(tensor) = tensor.from_floats(of: [1., 2., 3.], into: d1)
  tensor
  |> tensor.shape
  |> should.equal([3])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d2)
  tensor
  |> tensor.shape
  |> should.equal([3, 1])
}

pub fn reshape_test() {
  assert Ok(d0) = space.d0()
  assert Ok(d1) = space.d1(A)
  assert Ok(d2) = space.d2(#(A, -1), #(B, 1))
  assert Ok(d3) = space.d3(#(A, 1), #(B, -1), #(C, 1))
  assert Ok(d4) = space.d4(#(A, 1), #(B, 1), #(C, -1), #(D, 1))
  assert Ok(d5) = space.d5(#(A, 1), #(B, 1), #(C, 1), #(D, -1), #(E, 1))
  assert Ok(d6) =
    space.d6(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, -1), #(F, 1))

  assert Ok(tensor) =
    0.
    |> tensor.from_float
    |> tensor.reshape(into: d1)
  tensor
  |> tensor.shape
  |> should.equal([1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d2)
  tensor
  |> tensor.shape
  |> should.equal([1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d3)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d4)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d5)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d6)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d5)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d4)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d3)
  tensor
  |> tensor.shape
  |> should.equal([1, 1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d2)
  tensor
  |> tensor.shape
  |> should.equal([1, 1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d1)
  tensor
  |> tensor.shape
  |> should.equal([1])

  assert Ok(tensor) = tensor.reshape(put: tensor, into: d0)
  tensor
  |> tensor.shape
  |> should.equal([])
}

pub fn to_float_test() {
  0.
  |> tensor.from_float
  |> tensor.to_float
  |> should.equal(0.)

  0
  |> tensor.from_int
  |> tensor.as_format(apply: format.float32)
  |> tensor.to_float
  |> should.equal(0.)
}

pub fn to_int_test() {
  0
  |> tensor.from_int
  |> tensor.to_int
  |> should.equal(0)

  0.
  |> tensor.from_float
  |> tensor.as_format(apply: format.int32)
  |> tensor.to_int
  |> should.equal(0)
}

pub fn to_list_test() {
  let list = [1., 2., 3.]

  assert Ok(d1) = space.d1(A)
  assert Ok(d6) =
    space.d6(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1), #(F, -1))

  0.
  |> tensor.from_float
  |> tensor.to_list
  |> should.equal([0.])

  assert Ok(tensor) = tensor.from_floats(of: list, into: d1)
  tensor
  |> tensor.to_list
  |> should.equal(list)

  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
  tensor
  |> tensor.to_list
  |> should.equal(list)
}

//pub fn to_lists_test() {
//  let list = [1., 2., 3.]
//  let lists = fn(list: a) {
//    list
//    |> dynamic.from
//    |> Lists
//  }
//
//  assert Ok(d1) = space.d1(A)
//  assert Ok(d2) = space.d2(#(A, -1), #(B, 1))
//  assert Ok(d6) =
//    space.d6(#(A, 1), #(B, 1), #(C, 1), #(D, 1), #(E, 1), #(F, -1))
//
//  0.
//  |> tensor.from_float
//  |> tensor.to_lists
//  |> should.equal(lists([0.]))
//
//  assert Ok(tensor) = tensor.from_floats(of: list, into: d1)
//  tensor
//  |> tensor.to_lists
//  |> should.equal(lists(list))
//
//  assert Ok(tensor) = tensor.from_floats(of: list, into: d2)
//  tensor
//  |> tensor.to_lists
//  |> should.equal(lists([[1.], [2.], [3.]]))
//
//  assert Ok(tensor) = tensor.from_floats(of: list, into: d6)
//  tensor
//  |> tensor.to_lists
//  |> should.equal(lists([[[[[[1., 2., 3.]]]]]]))
//}

pub fn to_native_test() {
  let native =
    [1, 2, 3, 4, 5, 6, 7, 8]
    |> dynamic.from
    |> erlang_tensor
  assert Ok(space) = space.d3(#(A, 2), #(B, -1), #(C, 2))
  assert Ok(tensor) =
    tensor.from_native(of: native, into: space, with: format.int32)
  tensor
  |> tensor.to_native
  |> erlang_shape
  |> should.equal(dynamic.from(#(2, 2, 2)))
}
