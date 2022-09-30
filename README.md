# argamak 🐎

[![Hex Package](https://img.shields.io/hexpm/v/argamak?color=ffaff3&label=%F0%9F%93%A6)](https://hex.pm/packages/argamak)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3?label=%F0%9F%93%9A)](https://hexdocs.pm/argamak/)
[![License](https://img.shields.io/hexpm/l/argamak?color=ffaff3&label=%F0%9F%93%83)](https://github.com/tynanbe/argamak/blob/main/LICENSE)
[![Build](https://img.shields.io/github/workflow/status/tynanbe/argamak/CI?color=ffaff3&label=%E2%9C%A8)](https://github.com/tynanbe/argamak/actions)

A Gleam library for tensor maths.

> “I admire the elegance of your method of computation; it must be nice to ride
> through these fields upon the horse of true mathematics while the like of us
> have to make our way laboriously on foot.”
>
> —Albert Einstein, to Tullio Levi-Civita, circa 1915–1917

<p align="center" width="100%"><a href="https://www.wikiwand.com/en/Akhal-Teke"><img alt="Argamak: A shiny steed." src="https://github.com/tynanbe/argamak/raw/main/argamak.jpg" width="250"></a></p>

## Installation

### As a dependency of your Gleam project

• Add `argamak` to `gleam.toml` from the command line

```shell
$ gleam add argamak
```

### As a dependency of your Mix project

• Add `argamak` to `mix.exs`

```elixir
defp deps do
  [
    {:argamak, "~> 0.2"},
  ]
end
```

### As a dependency of your Rebar3 project

• Add `argamak` to `rebar.config`

```erlang
{deps, [
  {argamak, "0.2.0"}
]}.
```

### JavaScript

The `@tensorflow/tfjs` package is a runtime requirement for `argamak`, and its
import path in the `argamak_ffi.mjs` module might need adjustment, depending on
your use case.

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
