# Changelog

## v0.3.0 - 2022-11-20

- The `tensor` module gains the `TensorResult` type; the `from_bool` and
  `from_bools` creation functions; the `size` reflection function; the `squeeze`
  transformation function; the `equal`, `not_equal`, `greater`,
  `greater_or_equal`, `less`, `less_or_equal`, `logical_and`, `logical_or`,
  `logical_xor`, and `logical_not` logical functions; the `add`, `subtract`,
  `multiply`, `divide`, `try_divide`, `remainder`, `try_remainder`, `modulo`,
  `try_modulo`, `power`, `max`, and `min` arithmetic functions; the
  `absolute_value`, `negate`, `sign`, `ceiling`, `floor`, `round`, `exp`,
  `square_root`, and `ln` basic math functions; the `all`, `in_situ_all`, `any`,
  `in_situ_any`, `arg_max`, `in_situ_arg_max`, `arg_min`, `in_situ_arg_min`,
  `max_over`, `in_situ_max_over`, `min_over`, `in_situ_min_over`, `sum`,
  `in_situ_sum`, `product`, `in_situ_product`, `mean`, and `in_situ_mean`
  reduction functions; the `to_bool`, `to_floats`, `to_ints`, and `to_bools`
  conversion functions; and the `debug` and `print_data` utility functions.
- The `tensor` module's `as_format` function has been renamed to `reformat` and
  now takes a `Format` record instead of a function reference.
- The `tensor` module no longer includes the `to_list` function.
- The `Tensor` type signature now includes only the numeric format as a generic.
- The `axis` module has been added with the `Axis` and `Axes` types; the `name`
  and `size` reflection functions; and the `rename` and `resize` transformation
  functions.
- The `space` module gains the `from_list` creation function; and the `map` and
  `merge` transformation functions.
- The `space` module no longer includes the `elements` and `map_elements`
  functions.
- The `space` module's `d0` function has been renamed to `new` and noi returns
  an empty `Space` record directly.
- The `space` module and its `Space` and `SpaceError` types have been reworked:
  The `Space` type signature no longer includes any generics, and the
  constructors `D0` through `D6` have been removed.
- The `Format` type has been reworked and now includes the numeric format as a
- Several numeric format types have been added to the `format` module.
  generic.
- The `util` module has been removed.

## v0.2.0 - 2022-09-29

- The `space` module gets an updated `d1` function so the dimension size can be
  given.
- The `tensor` module gains the `broadcast`, and `broadcast_over` functions for
  all compilation targets.
- Argamak now compiles and runs with the JavaScript target.
- Argamak now uses the `gleam` build tool.

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
