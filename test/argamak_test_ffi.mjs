import { tensor as tf_tensor } from "@tensorflow/tfjs-node";

export const tensor = (x) => tf_tensor(eval(x.inspect()));

export const shape = (x) => x.shape;

export const type = (x) => x.dtype;

export const infinity = () => Infinity;

export const neg_infinity = () => -Infinity;
