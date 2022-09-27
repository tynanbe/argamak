import gleam/result

if erlang {
  import gleam/dynamic
  import gleam/erlang/atom
  import gleam/list
  import gleam/string
}

if javascript {
  import gleam/dynamic.{Dynamic}
}

/// An error type for utility functions.
pub type UtilError {
  InvalidRecord
}

/// Results in a `String` converted from a custom type record's name on success,
/// or a `UtilError` on failure.
///
/// ## Examples
///
/// ```gleam
/// > type Axis { X }
/// > record_to_string(X)
/// Ok("X")
///
/// > record_to_string(3)
/// Error(InvalidRecord)
/// ```
///
pub fn record_to_string(record: a) -> Result(String, UtilError) {
  do_record_to_string(record)
}

if erlang {
  fn do_record_to_string(record: a) -> Result(String, UtilError) {
    try atom =
      record
      |> dynamic.from
      |> dynamic.any(of: [
        atom.from_dynamic,
        dynamic.element(at: 0, of: atom.from_dynamic),
      ])
      |> result.replace_error(InvalidRecord)

    let atom =
      atom
      |> atom.to_string
      |> string.split(on: "_")
      |> list.flat_map(with: fn(string) {
        assert Ok(#(grapheme, string)) = string.pop_grapheme(string)
        [string.uppercase(grapheme), string]
      })
      |> string.concat

    Ok(atom)
  }
}

if javascript {
  fn do_record_to_string(record: a) -> Result(String, UtilError) {
    // TODO promise and rescue
    record
    |> prototype
    |> get("constructor")
    |> get("name")
    |> dynamic.string
    |> result.replace_error(InvalidRecord)
  }

  external fn get(a, String) -> Dynamic =
    "" "Reflect.get"

  external fn prototype(a) -> Dynamic =
    "" "Reflect.getPrototypeOf"
}
