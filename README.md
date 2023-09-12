# argamak 🐎

[![Hex Package](https://img.shields.io/hexpm/v/argamak?color=ffaff3&label&labelColor=2f2f2f&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsPSIjZmVmZWZjIiBkPSJNIDYuMjgzMiwxLjU5OTYgOS4yODMyLDYuNzk0OSBIIDE0LjcwNTEgTCAxNy43MDUxLDEuNTk5NiBaIE0gMTguMTQwNywxLjg0MzggbCAtMyw1LjE5NzMgMi43MTQ5LDQuNjk5MiBoIDYgeiBNIDUuODUzNSwxLjg1NTUgMC4xNDQ1LDExLjc0MDIgSCA2LjE0NDUgTCA4Ljg1MTYsNy4wNDg4IFogTSAwLjE0NDUsMTIuMjQwMiA1Ljg1MzUsMjIuMTI3IDguODUxNiwxNi45MzM2IDYuMTQ0NSwxMi4yNDAyIFogbSAxNy43MTEsMCAtMi43MTQ5LDQuNzAxMiAzLDUuMTk1MyA1LjcxNDksLTkuODk2NSB6IE0gOS4yODMyLDE3LjE4NzUgNi4yODUyLDIyLjM4MDkgSCAxNy43MDMyIEwgMTQuNzA1MSwxNy4xODc1IFoiLz48L3N2Zz4K)](https://hex.pm/packages/argamak)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3?label&labelColor=2f2f2f&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNiAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsPSIjZmVmZWZjIiBkPSJNMjUuNjA5IDcuNDY5YzAuMzkxIDAuNTYyIDAuNSAxLjI5NyAwLjI4MSAyLjAxNmwtNC4yOTcgMTQuMTU2Yy0wLjM5MSAxLjMyOC0xLjc2NiAyLjM1OS0zLjEwOSAyLjM1OWgtMTQuNDIyYy0xLjU5NCAwLTMuMjk3LTEuMjY2LTMuODc1LTIuODkxLTAuMjUtMC43MDMtMC4yNS0xLjM5MS0wLjAzMS0xLjk4NCAwLjAzMS0wLjMxMyAwLjA5NC0wLjYyNSAwLjEwOS0xIDAuMDE2LTAuMjUtMC4xMjUtMC40NTMtMC4wOTQtMC42NDEgMC4wNjMtMC4zNzUgMC4zOTEtMC42NDEgMC42NDEtMS4wNjIgMC40NjktMC43ODEgMS0yLjA0NyAxLjE3Mi0yLjg1OSAwLjA3OC0wLjI5Ny0wLjA3OC0wLjY0MSAwLTAuOTA2IDAuMDc4LTAuMjk3IDAuMzc1LTAuNTE2IDAuNTMxLTAuNzk3IDAuNDIyLTAuNzE5IDAuOTY5LTIuMTA5IDEuMDQ3LTIuODQ0IDAuMDMxLTAuMzI4LTAuMTI1LTAuNjg4LTAuMDMxLTAuOTM4IDAuMTA5LTAuMzU5IDAuNDUzLTAuNTE2IDAuNjg4LTAuODI4IDAuMzc1LTAuNTE2IDEtMiAxLjA5NC0yLjgyOCAwLjAzMS0wLjI2Ni0wLjEyNS0wLjUzMS0wLjA3OC0wLjgxMiAwLjA2My0wLjI5NyAwLjQzOC0wLjYwOSAwLjY4OC0wLjk2OSAwLjY1Ni0wLjk2OSAwLjc4MS0zLjEwOSAyLjc2Ni0yLjU0N2wtMC4wMTYgMC4wNDdjMC4yNjYtMC4wNjMgMC41MzEtMC4xNDEgMC43OTctMC4xNDFoMTEuODkxYzAuNzM0IDAgMS4zOTEgMC4zMjggMS43ODEgMC44NzUgMC40MDYgMC41NjIgMC41IDEuMjk3IDAuMjgxIDIuMDMxbC00LjI4MSAxNC4xNTZjLTAuNzM0IDIuNDA2LTEuMTQxIDIuOTM4LTMuMTI1IDIuOTM4aC0xMy41NzhjLTAuMjAzIDAtMC40NTMgMC4wNDctMC41OTQgMC4yMzQtMC4xMjUgMC4xODctMC4xNDEgMC4zMjgtMC4wMTYgMC42NzIgMC4zMTMgMC45MDYgMS4zOTEgMS4wOTQgMi4yNSAxLjA5NGgxNC40MjJjMC41NzggMCAxLjI1LTAuMzI4IDEuNDIyLTAuODkxbDQuNjg4LTE1LjQyMmMwLjA5NC0wLjI5NyAwLjA5NC0wLjYwOSAwLjA3OC0wLjg5MSAwLjM1OSAwLjE0MSAwLjY4OCAwLjM1OSAwLjkyMiAwLjY3MnpNOC45ODQgNy41Yy0wLjA5NCAwLjI4MSAwLjA2MyAwLjUgMC4zNDQgMC41aDkuNWMwLjI2NiAwIDAuNTYyLTAuMjE5IDAuNjU2LTAuNWwwLjMyOC0xYzAuMDk0LTAuMjgxLTAuMDYzLTAuNS0wLjM0NC0wLjVoLTkuNWMtMC4yNjYgMC0wLjU2MiAwLjIxOS0wLjY1NiAwLjV6TTcuNjg4IDExLjVjLTAuMDk0IDAuMjgxIDAuMDYzIDAuNSAwLjM0NCAwLjVoOS41YzAuMjY2IDAgMC41NjItMC4yMTkgMC42NTYtMC41bDAuMzI4LTFjMC4wOTQtMC4yODEtMC4wNjMtMC41LTAuMzQ0LTAuNWgtOS41Yy0wLjI2NiAwLTAuNTYyIDAuMjE5LTAuNjU2IDAuNXoiPjwvcGF0aD48L3N2Zz4K)](https://hexdocs.pm/argamak/)
[![License](https://img.shields.io/hexpm/l/argamak?color=ffaff3&label&labelColor=2f2f2f&logo=data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgd2lkdGg9IjM0IiBoZWlnaHQ9IjI4IiB2aWV3Qm94PSIwIDAgMzQgMjgiPgo8cGF0aCBmaWxsPSIjZmVmZWZjIiBkPSJNMjcgN2wtNiAxMWgxMnpNNyA3bC02IDExaDEyek0xOS44MjggNGMtMC4yOTcgMC44NDQtMC45ODQgMS41MzEtMS44MjggMS44Mjh2MjAuMTcyaDkuNWMwLjI4MSAwIDAuNSAwLjIxOSAwLjUgMC41djFjMCAwLjI4MS0wLjIxOSAwLjUtMC41IDAuNWgtMjFjLTAuMjgxIDAtMC41LTAuMjE5LTAuNS0wLjV2LTFjMC0wLjI4MSAwLjIxOS0wLjUgMC41LTAuNWg5LjV2LTIwLjE3MmMtMC44NDQtMC4yOTctMS41MzEtMC45ODQtMS44MjgtMS44MjhoLTcuNjcyYy0wLjI4MSAwLTAuNS0wLjIxOS0wLjUtMC41di0xYzAtMC4yODEgMC4yMTktMC41IDAuNS0wLjVoNy42NzJjMC40MjItMS4xNzIgMS41MTYtMiAyLjgyOC0yczIuNDA2IDAuODI4IDIuODI4IDJoNy42NzJjMC4yODEgMCAwLjUgMC4yMTkgMC41IDAuNXYxYzAgMC4yODEtMC4yMTkgMC41LTAuNSAwLjVoLTcuNjcyek0xNyA0LjI1YzAuNjg4IDAgMS4yNS0wLjU2MiAxLjI1LTEuMjVzLTAuNTYyLTEuMjUtMS4yNS0xLjI1LTEuMjUgMC41NjItMS4yNSAxLjI1IDAuNTYyIDEuMjUgMS4yNSAxLjI1ek0zNCAxOGMwIDMuMjE5LTQuNDUzIDQuNS03IDQuNXMtNy0xLjI4MS03LTQuNXYwYzAtMC42MDkgNS40NTMtMTAuMjY2IDYuMTI1LTExLjQ4NCAwLjE3Mi0wLjMxMyAwLjUxNi0wLjUxNiAwLjg3NS0wLjUxNnMwLjcwMyAwLjIwMyAwLjg3NSAwLjUxNmMwLjY3MiAxLjIxOSA2LjEyNSAxMC44NzUgNi4xMjUgMTEuNDg0djB6TTE0IDE4YzAgMy4yMTktNC40NTMgNC41LTcgNC41cy03LTEuMjgxLTctNC41djBjMC0wLjYwOSA1LjQ1My0xMC4yNjYgNi4xMjUtMTEuNDg0IDAuMTcyLTAuMzEzIDAuNTE2LTAuNTE2IDAuODc1LTAuNTE2czAuNzAzIDAuMjAzIDAuODc1IDAuNTE2YzAuNjcyIDEuMjE5IDYuMTI1IDEwLjg3NSA2LjEyNSAxMS40ODR6Ij48L3BhdGg+Cjwvc3ZnPgo=)](https://github.com/tynanbe/argamak/blob/main/LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/tynanbe/argamak/ci.yml?branch=main&color=ffaff3&label&labelColor=2f2f2f&logo=github-actions&logoColor=fefefc)](https://github.com/tynanbe/argamak/actions)

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

The `@tensorflow/tfjs` package is a runtime requirement for `argamak`; however,
its import path in the `argamak_ffi.mjs` module might need adjustment,
depending on your use case. It can be used as is in your Node.js project after
running `npm install @tensorflow/tfjs-node` or an equivalent command.

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
  let assert Ok(d2) =
    space.d2(Infer(name: "Horse"), Axis(name: "Trial", size: 2))

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
  use x <- result.try(tensor.from_floats(of: times, into: d2))

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
  use horse <- result.try(
    mean_times
    |> tensor.arg_min(with: all_axes)
    |> tensor.debug
    |> tensor.to_int,
  )
  use horse <- result.try(
    horses
    |> list.at(get: horse)
    |> result.replace_error(InvalidData),
  )

  // Finally, we make our announcement!
  //
  announce(horse <> " wins the day with a mean time of " <> time <> " minutes!")
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
