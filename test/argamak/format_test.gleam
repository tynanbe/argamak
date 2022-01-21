import argamak/format
import gleeunit/should

pub fn to_string_test() {
  format.float32()
  |> format.to_string
  |> should.equal("Float32")

  format.int32()
  |> format.to_string
  |> should.equal("Int32")
}
