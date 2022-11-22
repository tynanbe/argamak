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
    {:argamak, "~> 0.3"},
  ]
end
```

### As a dependency of your Rebar3 project

• Add `argamak` to `rebar.config`

```erlang
{deps, [
  {argamak, "0.3.0"}
]}.
```

### JavaScript

The `@tensorflow/tfjs` package is a runtime requirement for `argamak`; its
import path in the `argamak_ffi.mjs` module might need adjustment, depending on
your use case.

## Usage

```gleam
// derby.gleam
import argamak/axis.{Axis, Infer}
import argamak/space
import argamak/tensor.{InvalidData, TensorError}
import gleam/function
import gleam/io
import gleam/list
import gleam/result
import gleam/string

pub fn announce_winner(
  from horses: List(String),
  with times: List(Float),
) -> Result(Nil, TensorError) {
  // Space records help maintain a clear understanding of a Tensor's data.
  //
  // We begin by creating a two-dimensional Space with "Horse" and "Trial" Axes.
  // The "Trial" Axis size is two because horses always run twice in our derby.
  // The "Horse" Axis size will be inferred based on the data when a Tensor is
  // put into our Space (perhaps we won't always know how many horses will run).
  //
  assert Ok(d2) = space.d2(Infer(name: "Horse"), Axis(name: "Trial", size: 2))

  // Every Tensor has a numerical Format, a Space, and some data.
  // A 2d Tensor can be visualized like a table or matrix.
  //
  // Tensor(
  //   Format(Float32)
  //   Space(Axis("Horse", 5), Axis("Trial", 2))
  //
  //                Trial
  //  H [[horse1_time1, horse1_time2],
  //  o  [horse2_time1, horse2_time2],
  //  r  [horse3_time1, horse3_time2],
  //  s  [horse4_time1, horse4_time2],
  //  e  [horse5_time1, horse5_time2]],
  // )
  //
  // Next we create a Tensor from a List of times and put it into our 2d Space.
  //
  try x = tensor.from_floats(of: times, into: d2)

  let announce = function.compose(string.inspect, io.println)

  announce("Trial times per horse")
  tensor.print(x)

  // Axes can be referenced by name.
  //
  // Here we reduce away the "Trial" Axis to get each horse's mean run time.
  //
  announce("Mean time per horse")
  let mean_times =
    x
    |> tensor.mean(with: fn(a) { axis.name(a) == "Trial" })
    |> tensor.debug

  // This catch-all function will reduce away all Axes, although at this point
  // only the "Horse" Axis remains.
  //
  let all_axes = fn(_) { True }

  // We get a String representation of the minimum mean time.
  //
  announce("Fastest mean time")
  let time =
    mean_times
    |> tensor.min_over(with: all_axes)
    |> tensor.debug
    |> tensor.to_string(return: tensor.Data, wrap_at: 0)

  // And we get an index number, followed by the name of the winning horse.
  //
  announce("Fastest horse")
  try horse =
    mean_times
    |> tensor.arg_min(with: all_axes)
    |> tensor.debug
    |> tensor.to_int
  try horse =
    horses
    |> list.at(get: horse)
    |> result.replace_error(InvalidData)

  // Finally, we make our announcement!
  //
  horse <> " wins the day with a mean time of " <> time <> " minutes!"
  |> announce
  |> Ok
}
```

### Example

```gleam
> derby.announce_winner(
>   from: ["Pony Express", "Hay Girl", "Low Rider"],
>   with: [1.2, 1.3, 1.3, 1.0, 1.5, 0.9],
> )
"Trial times per horse"
Tensor(
  Format(Float32),
  Space(Axis("Horse", 3), Axis("Trial", 2)),
  [[1.2, 1.3],
   [1.3, 1.0],
   [1.5, 0.9]],
)
"Mean time per horse"
Tensor(
  Format(Float32),
  Space(Axis("Horse", 3)),
  [1.25, 1.15,  1.2],
)
"Fastest mean time"
Tensor(
  Format(Float32),
  Space(),
  1.15,
)
"Fastest horse"
Tensor(
  Format(Float32),
  Space(),
  1.0,
)
"Hay Girl wins the day with a mean time of 1.15 minutes!"
Ok(Nil)
```
