import * as tf from "@tensorflow/tfjs-node";
import { Error as GleamError, List, Ok, Result, toList } from "./gleam.mjs";
import {
  CannotBroadcast,
  IncompatibleShape,
  InvalidData,
} from "./argamak/tensor.mjs";

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Constants                              //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

const Nil = undefined;

const Tensor = tf.Tensor.prototype;

const Extrema = {
  int32: { min: -2_147_483_648, max: 2_147_483_647 },
  float32: {
    min: -340_282_346_638_528_859_811_704_183_484_516_925_440,
    max: 340_282_346_638_528_859_811_704_183_484_516_925_440,
  },
};

class Fn {
  constructor(f) {
    this.f = f;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // Tensor Methods                       //
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  static clip_reformat_like(x) {
    return x.dtype === this.dtype
      // Don't clip if reformat is noop.
      ? this
      : this.reformat_like(x).clip_based_on(this);
  }

  static clip_reformat(x) {
    return (x === this.dtype ? this : this.cast(x)).clip_based_on(this);
  }

  static clip_based_on(x) {
    let format = this.dtype;

    let scalar = (x) => tf.scalar(Extrema[format][x], format);

    let min = scalar("min");
    let max = scalar("max");

    let less = x.less(min);
    let greater = x.greater(max);

    Tensor.replace = function (predicate, value) {
      return value.where(predicate, this);
    };

    return this.replace(less, min).replace(greater, max);
  }

  static reformat_like(x) {
    x = x.dtype;
    return x === this.dtype ? this : this.cast(x);
  }

  static to_number() {
    if (Number.isFinite(this)) {
      return this;
    } else if (tf.util.isScalarShape(this.shape)) {
      return this.arraySync();
    }
    throw new Error(Nil);
  }

  static to_flat_list() {
    return toList(this.reshape([-1]).arraySync());
  }

  static add_fn(x) {
    this[x] = Fn[x];
    return this;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // Result Methods                       //
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  static map(f) {
    return this.isOk() ? new Ok(f(this[0])) : this;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // Fn Methods                           //
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  broadcast_result() {
    return this.result(CannotBroadcast);
  }

  shape_result() {
    return this.result(IncompatibleShape);
  }

  result(error_type = InvalidData) {
    try {
      let x = this.f();
      // Detect any NaN
      return tf.equal(x, x).all().arraySync()
        ? new Ok(x)
        : new GleamError(new InvalidData());
    } catch {
      return new GleamError(new error_type());
    }
  }
}

Tensor.add_fn = Fn.add_fn;

Tensor.add_fn("clip_reformat_like")
  .add_fn("clip_reformat")
  .add_fn("clip_based_on")
  .add_fn("reformat_like")
  .add_fn("to_number")
  .add_fn("to_flat_list");

Result.prototype.map = Fn.map;

const fn = (f) => new Fn(f);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Creation Functions              //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const tensor = (x, format) =>
  fn(() => {
    if (List.isList(x)) {
      x = x.toArray();
      if (!x.length) {
        throw new Error(Nil);
      }
    }
    let shape = Array.isArray(x) ? [x.length] : [];
    return tf.tensor(x, shape, format);
  }).result();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Reflection Functions            //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const size = (x) => x.size;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Transformation Functions        //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const reformat = (x, format) => format === x.dtype ? x : x.cast(format);

export const reshape = (x, shape) =>
  fn(() => x.reshape(shape.toArray())).shape_result();

export const broadcast = (x, shape) =>
  fn(() => x.broadcastTo(shape.toArray())).shape_result();

export const squeeze = (x, i) => x.squeeze(i.toArray());

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Logical Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const equal = (a, b) =>
  fn(() => a.equal(b).reformat_like(a)).broadcast_result();

export const not_equal = (a, b) =>
  fn(() => a.notEqual(b).reformat_like(a)).broadcast_result();

export const greater = (a, b) =>
  fn(() => a.greater(b).reformat_like(a)).broadcast_result();

export const greater_or_equal = (a, b) =>
  fn(() => a.greaterEqual(b).reformat_like(a)).broadcast_result();

export const less = (a, b) =>
  fn(() => a.less(b).reformat_like(a)).broadcast_result();

export const less_or_equal = (a, b) =>
  fn(() => a.lessEqual(b).reformat_like(a)).broadcast_result();

export const logical_and = (a, b) =>
  fn(() => a.cast("bool").logicalAnd(b.cast("bool")).reformat_like(a))
    .broadcast_result();

export const logical_or = (a, b) =>
  fn(() => a.cast("bool").logicalOr(b.cast("bool")).reformat_like(a))
    .broadcast_result();

export const logical_xor = (a, b) =>
  fn(() => a.cast("bool").logicalXor(b.cast("bool")).reformat_like(a))
    .broadcast_result();

export const logical_not = (x) => x.cast("bool").logicalNot().reformat_like(x);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Arithmetic Functions            //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const add = (a, b) => fn(() => a.add(b)).broadcast_result();

export const subtract = (a, b) => fn(() => a.sub(b)).broadcast_result();

export const multiply = (a, b) => fn(() => a.mul(b)).broadcast_result();

export const divide = (a, b) =>
  fn(() => a.div(b))
    .broadcast_result()
    .map((x) => x.clip_reformat_like(a));

export const modulo = (a, b) => fn(() => a.mod(b)).broadcast_result();

export const power = (a, b) => fn(() => a.pow(b)).broadcast_result();

export const max = (a, b) => fn(() => a.maximum(b)).broadcast_result();

export const min = (a, b) => fn(() => a.minimum(b)).broadcast_result();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Basic Math Functions            //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const absolute_value = tf.abs;

export const negate = tf.neg;

export const sign = tf.sign;

export const ceiling = (x) => x.cast("float32").ceil().reformat_like(x);

export const floor = (x) => x.cast("float32").floor().reformat_like(x);

export const round = tf.round;

export const exp = (x) => x.exp().clip_reformat_like(x);

export const square_root = (x) =>
  fn(() => x.cast("float32").sqrt())
    .result()
    .map((y) => y.clip_reformat_like(x));

export const ln = (x) =>
  fn(() => x.cast("float32").log())
    .result()
    .map((y) => y.clip_reformat_like(x));

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Reduction Functions             //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const all = (x, i) => x.cast("bool").all(i.toArray()).reformat_like(x);

export const any = (x, i) => x.cast("bool").any(i.toArray()).reformat_like(x);

export const arg_max = (x, i) => x.argMax(i).reformat_like(x);

export const arg_min = (x, i) => x.argMin(i).reformat_like(x);

export const max_over = (x, i) => x.max(i.toArray());

export const min_over = (x, i) => x.min(i.toArray());

export const sum = (x, i) => x.sum(i.toArray()).clip_reformat_like(x);

export const product = (x, i) => x.prod(i.toArray()).clip_reformat_like(x);

export const mean = (x, i) => x.mean(i.toArray()).reformat_like(x);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Slicing & Joining Functions     //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const concat = (xs, i) => tf.concat(xs.toArray(), i);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Conversion Functions            //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const to_float = (x) =>
  fn(() => x.clip_reformat("float32").to_number()).shape_result();

export const to_int = (x) =>
  fn(() => x.clip_reformat("int32").to_number()).shape_result();

export const to_floats = (x) => x.clip_reformat("float32").to_flat_list();

export const to_ints = (x) => x.clip_reformat("int32").to_flat_list();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Tensor Utility Functions               //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export function prepare_to_string(x) {
  let is_int = "int32" === x.dtype;

  let [xs, item_width] = x
    .reshape([-1])
    .arraySync()
    .reduce(
      ([xs, item_width], x) => {
        x = Number(x).toFixed(3);
        x = x.includes("e") ? x : trim_trailing(x, "0");
        if (is_int) {
          x = trim_trailing(x, ".");
        } else {
          x = x.substr(-1) === "." ? `${x}0` : x;
        }
        return [[...xs, x], Math.max(x.length, item_width)];
      },
      [[], 0],
    );

  return [toList(xs), item_width];
}

export function columns() {
  let stdout = process.stdout;
  return stdout.isTTY ? stdout.columns : 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Format Functions                       //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

export const format_to_native = (x) => x.inspect().toLowerCase();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Private Functions                      //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

function trim_trailing(x, character) {
  x = `${x}`;
  let i = x.length;
  while (x.charAt(--i) === character) {
    x = x.slice(0, i);
  }
  return x;
}
