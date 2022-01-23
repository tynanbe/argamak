# Changelog

## v0.2.0

- The `space` module gets an updated `d1` function so the dimension size can be
  given.
- The `tensor` module gains the `broadcast`, and `broadcast_over` functions for
  all compilation targets (JavaScript planned).

## v0.1.0 - 2022-01-20

- Initial release!
- The `format` module gains the `Format` and `Native` (JavaScript planned) types
  for all compilation targets, along with the `float32`, `int32`, `to_native`,
  and `to_string` functions for all compilation targets (JavaScript planned) and
  the `bfloat16`, `float16`, `float64`, `int16`, `int64`, `int8`, `uint16`,
  `uint32`, `uint64`, and `uint8` functions for the Erlang compilation target.
- The `space` module gains the `D0`, `D1`, `D2`, `D3`, `D4`, `D5`, `D6`,
  `Space`, and `SpaceError` (with `SpaceErrors` alias) types for all compilation
  targets, along with the `axes`, `d0`, `d1`, `d2`, `d3`, `d4`, `d5`, `d6`,
  `degree`, `elements`, `map_elements`, `shape`, and `to_string` functions for
  all compilation targets (JavaScript planned).
- The `tensor` module gains the `Native` (JavaScript planned), `Tensor`, and
  `TensorError` types for all compilation targets, along with the `as_format`,
  `axes`, `format`, `from_float`, `from_floats`, `from_int`, `from_ints`,
  `from_native`, `print`, `rank`, `reshape`, `shape`, `space`, `to_float`,
  `to_int`, `to_list`, and `to_native` functions for all compilation targets
  (JavaScript planned).
- The `util` module gains the `UtilError` type for all compilation targets,
  along with the `record_to_string` function for all compilation targets.
