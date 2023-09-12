defmodule :argamak_ffi do
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Constants                              #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  @result :gleam@result

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Creation Functions              #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def tensor(x, format), do: fn -> Nx.tensor(x, type: format) end |> result

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Reflection Functions            #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def size(x), do: Nx.size(x)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Transformation Functions        #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def reformat(x, like: y), do: Nx.as_type(x, Nx.type(y))
  def reformat(x, format), do: Nx.as_type(x, format)

  def reshape(x, shape),
    do: fn -> Nx.reshape(x, :erlang.list_to_tuple(shape)) end |> shape_result

  def broadcast(x, shape),
    do: fn -> Nx.broadcast(x, :erlang.list_to_tuple(shape)) end |> shape_result

  def squeeze(x, i), do: Nx.squeeze(x, axes: i)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Logical Functions               #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def equal(a, b),
    do: fn -> Nx.equal(a, b) |> reformat(like: a) end |> broadcast_result

  def not_equal(a, b),
    do: fn -> Nx.not_equal(a, b) |> reformat(like: a) end |> broadcast_result

  def greater(a, b),
    do: fn -> Nx.greater(a, b) |> reformat(like: a) end |> broadcast_result

  def greater_or_equal(a, b),
    do: fn -> Nx.greater_equal(a, b) |> reformat(like: a) end |> broadcast_result

  def less(a, b),
    do: fn -> Nx.less(a, b) |> reformat(like: a) end |> broadcast_result

  def less_or_equal(a, b),
    do: fn -> Nx.less_equal(a, b) |> reformat(like: a) end |> broadcast_result

  def logical_and(a, b),
    do: fn -> Nx.logical_and(a, b) |> reformat(like: a) end |> broadcast_result

  def logical_or(a, b),
    do: fn -> Nx.logical_or(a, b) |> reformat(like: a) end |> broadcast_result

  def logical_xor(a, b),
    do: fn -> Nx.logical_xor(a, b) |> reformat(like: a) end |> broadcast_result

  def logical_not(x), do: Nx.logical_not(x) |> reformat(like: x)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Arithmetic Functions            #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def add(a, b), do: fn -> Nx.add(a, b) end |> broadcast_result

  def subtract(a, b), do: fn -> Nx.subtract(a, b) end |> broadcast_result

  def multiply(a, b), do: fn -> Nx.multiply(a, b) end |> broadcast_result

  def divide(a, b),
    do: fn -> Nx.divide(a, b) end |> broadcast_result |> @result.map(&clip_reformat(&1, like: a))

  def remainder(a, b), do: fn -> Nx.remainder(a, b) end |> broadcast_result

  def power(a, b), do: fn -> Nx.pow(a, b) end |> broadcast_result

  def max(a, b), do: fn -> Nx.max(a, b) end |> broadcast_result

  def min(a, b), do: fn -> Nx.min(a, b) end |> broadcast_result

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Basic Math Functions            #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def absolute_value(x), do: Nx.abs(x)

  def negate(x), do: Nx.negate(x)

  def sign(x), do: Nx.sign(x)

  def ceiling(x), do: Nx.ceil(x)

  def floor(x), do: Nx.floor(x)

  def round(x), do: Nx.round(x)

  def exp(x), do: Nx.exp(x) |> clip_reformat(like: x)

  def square_root(x),
    do: fn -> Nx.sqrt(x) end |> result |> @result.map(&clip_reformat(&1, like: x))

  def ln(x),
    do: fn -> Nx.log(x) end |> result |> @result.map(&clip_reformat(&1, like: x))

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Reduction Functions             #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def all(x, i), do: Nx.all(x, axes: i) |> reformat(like: x)

  def any(x, i), do: Nx.any(x, axes: i) |> reformat(like: x)

  def arg_max(x, i), do: Nx.argmax(x, axis: i) |> reformat(like: x)

  def arg_min(x, i), do: Nx.argmin(x, axis: i) |> reformat(like: x)

  def max_over(x, i), do: Nx.reduce_max(x, axes: i)

  def min_over(x, i), do: Nx.reduce_min(x, axes: i)

  def sum(x, i), do: Nx.sum(x, axes: i) |> clip_reformat(like: x)

  def product(x, i), do: Nx.product(x, axes: i) |> clip_reformat(like: x)

  def mean(x, i), do: Nx.mean(x, axes: i) |> reformat(like: x)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Slicing & Joining Functions     #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def concat(xs, i), do: Nx.concatenate(xs, axis: i)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Conversion Functions            #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def to_float(x),
    do: fn -> clip_reformat(x, float_format(x)) |> Nx.to_number() end |> shape_result

  def to_int(x),
    do: fn -> clip_reformat(x, int_format(x)) |> Nx.to_number() end |> shape_result

  def to_floats(x), do: clip_reformat(x, float_format(x)) |> Nx.to_flat_list()

  def to_ints(x), do: clip_reformat(x, int_format(x)) |> Nx.to_flat_list()

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Tensor Utility Functions               #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def prepare_to_string(x) do
    {format, _} = Nx.type(x)
    is_int = Enum.any?([:s, :u], fn x -> x == format end)

    x
    |> Nx.to_flat_list()
    |> Enum.reverse()
    |> Enum.reduce({[], 0}, fn x, {xs, item_width} ->
      x =
        case x do
          :infinity ->
            "Infinity"

          :neg_infinity ->
            "-Infinity"

          _else ->
            x =
              ~s(~.3#{if is_int, do: "f", else: "g"})
              |> :io_lib.format([x + 0.0])
              |> :erlang.list_to_binary()
              |> String.trim_trailing("e+0")

            x =
              if String.contains?(x, "e") do
                x
              else
                String.trim_trailing(x, "0")
              end

            if is_int do
              String.trim_trailing(x, ".")
            else
              String.replace_trailing(x, ".", ".0")
            end
        end

      {[x | xs], Kernel.max(String.length(x), item_width)}
    end)
  end

  def columns() do
    case :io.columns() do
      {:ok, columns} -> columns
      _else -> 0
    end
  end

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Format Functions                       #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  def format_to_native(x) do
    case x do
      :float64 -> {:f, 64}
      :int64 -> {:s, 64}
      :uint64 -> {:u, 64}
      :float32 -> {:f, 32}
      :int32 -> {:s, 32}
      :uint32 -> {:u, 32}
      :bfloat16 -> {:bf, 16}
      :float16 -> {:f, 16}
      :int16 -> {:s, 16}
      :uint16 -> {:u, 16}
      :int8 -> {:s, 8}
      :uint8 -> {:u, 8}
    end
  end

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Private Tensor Functions               #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  defp clip_reformat(x, like: y) do
    if Nx.type(x) == Nx.type(y) do
      # Don't clip if reformat is noop.
      x
    else
      x
      |> reformat(like: y)
      |> clip(based_on: x)
    end
  end

  defp clip_reformat(x, format) do
    x
    |> reformat(format)
    |> clip(based_on: x)
  end

  defp clip(x, based_on: y) do
    format = Nx.type(x)

    scalar = fn f ->
      format
      |> f.()
      |> Nx.from_binary(format)
      |> Nx.reshape({})
    end

    min = scalar.(&Nx.Type.min_finite_binary/1)
    max = scalar.(&Nx.Type.max_finite_binary/1)

    less = Nx.less(y, min)
    greater = Nx.greater(y, max)

    replace = fn x, predicate, with: value -> Nx.select(predicate, value, x) end

    x
    |> replace.(less, with: min)
    |> replace.(greater, with: max)
  end

  defp float_format(x), do: Nx.type(x) |> Nx.Type.to_floating()

  defp int_format(x) do
    format = Nx.type(x)
    if Nx.Type.integer?(format), do: format, else: {:s, 32}
  end

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
  # Private Result Functions               #
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

  defp broadcast_result(f), do: result(f, or: :cannot_broadcast)

  defp shape_result(f), do: result(f, or: :incompatible_shape)

  defp result(f, opts \\ []) do
    try do
      x = f.()

      case x |> Nx.is_nan() |> Nx.any() |> Nx.to_number() do
        0 -> {:ok, x}
        1 -> {:error, :invalid_data}
      end
    rescue
      ArithmeticError -> {:error, :invalid_data}
      _else -> {:error, Keyword.get(opts, :or, :invalid_data)}
    end
  end
end
