import * as tf from "@tensorflow/tfjs-node";
import { Error, List, Ok, toList } from "./gleam.mjs";

const Nil = undefined;

export function tensor(data) {
  // TODO: disallow empty?
  data = List.isList(data) ? data.toArray() : data;
  return tf.tensor(data);
}

export function as_type(tensor, dtype) {
  return tensor.cast(dtype);
}

export function broadcast(tensor, shape) {
  return tensor.broadcastTo(shape);
}

export function reshape(tensor, shape) {
  return tensor.reshape(shape);
}

export function to_number(tensor) {
  let [number] = to_flat_list(tensor);
  return number;
}

export function to_flat_list(tensor) {
  return toList(tensor.reshape([-1]).arraySync());
}

export function rescue(fun) {
  try {
    return new Ok(fun());
  } catch (error) {
    return new Error([error.name, error.message]);
  }
}

export function list_to_tuple(list) {
  return list.toArray();
}

export function format_to_native(format) {
  return format.inspect().toLowerCase();
}
