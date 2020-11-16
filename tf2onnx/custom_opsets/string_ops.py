# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
""" tf2onnx mapping functions for string ops using custom ops domain. """
import logging
from onnx import TensorProto
from tf2onnx import constants
from tf2onnx.handler import tf_op
from tf2onnx import utils
import numpy as np
from onnx import helper
from onnx.onnx_pb import TensorProto

logger = logging.getLogger(__name__)

@tf_op("StringSplit", domain=constants.STRING_OPS_DOMAIN)
class StringOps:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.STRING_OPS_DOMAIN
        skip_empty = node.get_attr_value('skip_empty', True)
        for a in list(node.attr.keys()):
            del node.attr[a]
        unsqueeze_node = ctx.make_node("Unsqueeze", [node.input[1]], attr={'axes': [0]})
        skip_empty_const = ctx.make_const(utils.make_name('skip_empty_const'), np.array([skip_empty], np.bool))
        ctx.replace_inputs(node, [node.input[0], unsqueeze_node.output[0], skip_empty_const.output[0]])
    
@tf_op("StringToHashBucketFast", domain=constants.STRING_OPS_DOMAIN)
class StringToHashBucketFast:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.STRING_OPS_DOMAIN
        num_buckets = node.get_attr_int('num_buckets')
        num_buckets_const = ctx.make_const(utils.make_name('num_buckets'), np.array([num_buckets], dtype=np.int64))
        ctx.replace_inputs(node, [node.input[0], num_buckets_const.output[0]])
        del node.attr['num_buckets']

def make_string_const(ctx, name, np_val):
    onnx_tensor = helper.make_tensor(name, TensorProto.STRING, np_val.shape, np_val, raw=False)
    dtype = onnx_tensor.data_type
    node = ctx.make_node("Const", [], outputs=[name], name=name, attr={"value": onnx_tensor},
                            skip_conversion=True, dtypes=[dtype], infer_shape_dtype=False)
    ctx.set_shape(name, np_val.shape)
    ctx.set_dtype(name, TensorProto.STRING)
    return node

@tf_op("StaticRegexReplace", domain=constants.STRING_OPS_DOMAIN)
class StaticRegexReplace:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.STRING_OPS_DOMAIN
        node.type = "StringRegexReplace"
        pattern = node.get_attr_value("pattern")
        rewrite = node.get_attr_value("rewrite")
        utils.make_sure(node.get_attr_value("replace_global") != 0,
                        "Can only convert StaticRegexReplace if replace_global is True")
        pattern_node = make_string_const(ctx, utils.make_name("pattern"), np.array([pattern], np.object))
        rewrite_node = make_string_const(ctx, utils.make_name("rewrite"), np.array([rewrite], np.object))
        del node.attr["pattern"]
        del node.attr["rewrite"]
        del node.attr["replace_global"]
        ctx.replace_inputs(node, [node.input[0], pattern_node.output[0], rewrite_node.output[0]])

@tf_op("StringJoin", domain=constants.STRING_OPS_DOMAIN)
class StringJoin:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        node.domain = constants.STRING_OPS_DOMAIN
        separator = node.get_attr_value("separator")
        if separator is None:
            separator = b''
        separator_node = make_string_const(ctx, utils.make_name("separator"), np.array([separator], np.object))
        axis_node = ctx.make_const(utils.make_name("axis"), np.array([0], np.int64))
        inps_with_shapes = [i for i in node.input if ctx.get_shape(i) != []]
        shape_node = None
        if 0 < len(inps_with_shapes) < len(node.input):
            shape_node = ctx.make_node("Shape", [inps_with_shapes[0]])
        unsqueezes = []
        for inp in node.input:
            if ctx.get_shape(inp) == [] and shape_node is not None:
                expand_node = ctx.make_node("Expand", [inp, shape_node.output[0]])
                inp = expand_node.output[0]
            unsqueeze_node = ctx.make_node("Unsqueeze", [inp], attr={'axes': [0]})
            unsqueezes.append(unsqueeze_node.output[0])
        stack_node = ctx.make_node("Concat", unsqueezes, attr={'axis': 0})
        ctx.replace_inputs(node, [stack_node.output[0], separator_node.output[0], axis_node.output[0]])