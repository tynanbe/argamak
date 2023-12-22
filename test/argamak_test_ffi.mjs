import { tensor as tf_tensor } from "@tensorflow/tfjs-node";
import { inspect } from "../gleam_stdlib/gleam_stdlib.mjs";

export const tensor = (x) => tf_tensor(eval(inspect(x)));

export const shape = (x) => x.shape;

export const type = (x) => x.dtype;

export const infinity = () => Infinity;

export const neg_infinity = () => -Infinity;
