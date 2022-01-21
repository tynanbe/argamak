# argamak 🐎

[![Hex Package](https://img.shields.io/hexpm/v/argamak?color=ffaff3&label=%F0%9F%93%A6)](https://hex.pm/packages/argamak)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3?label=%F0%9F%93%9A)](https://hexdocs.pm/argamak/)
[![License](https://img.shields.io/hexpm/l/argamak?color=ffaff3&label=%F0%9F%93%83)](https://hex.pm/packages/argamak)

A Gleam library for tensor maths.

> “I admire the elegance of your method of computation; it must be nice to ride
> through these fields upon the horse of true mathematics while the like of us
> have to make our way laboriously on foot.”
>
> —Albert Einstein, to Tullio Levi-Civita, circa 1915–1917

<p align="center" width="100%"><img alt="Argamak: A shiny steed." src="https://github.com/tynanbe/argamak/raw/main/argamak.jpg" width="250"></p>

## Installation

### Mix

```elixir
# mix.exs
defp deps do
  [
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx"},
    {:argamak, "~> 0.1"},
  ]
end
```

*For the Erlang compilation target, `argamak` depends on `Elixir.Nx` and, for
the time being, is easiest to use as a dependency of a `Mix` project.*

## Usage

```gleam
// derby.gleam
import argamak/space
import argamak/tensor

pub type Axis {
  Horse
  Trial
}

pub fn print_times(from list: List(Float)) {
  assert Ok(space) = space.d2(#(Horse, 3), #(Trial, -1))
  try tensor = tensor.from_floats(of: list, into: space)

  tensor
  |> tensor.print(wrap_at: -1, meta: True)
  |> Ok
}
```

### Example

```gleam
> derby.print_times(from: [1.2, 1.3, 1.3, 1., 1.5, 0.9])
// Tensor
// format: Float32
// space: D2 #(Horse, 3), #(Trial, 2)
// data:
// [[1.2, 1.3],
//  [1.3, 1.0],
//  [1.5, 0.9]]
Ok(Nil)
```
