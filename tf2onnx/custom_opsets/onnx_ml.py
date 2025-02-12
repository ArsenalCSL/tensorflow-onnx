# SPDX-License-Identifier: Apache-2.0

""" tf2onnx mapping functions for onnx ml domain. """
import logging
import numpy as np
from onnx import TensorProto
from onnx import numpy_helper
from tf2onnx import constants
from tf2onnx.handler import tf_op
from tf2onnx import utils


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring,unnecessary-pass

@tf_op("HashTableV2")
class HashTable:
    @classmethod
    def version_8(cls, ctx, node, **kwargs):
        """ HashTable will be removed """
        pass


@tf_op("LookupTableFindV2")
class LookupTableFind:
    @classmethod
    def version_8(cls, ctx, node, initialized_tables, **kwargs):
        """ convert lookup to category mapper """
        table_node = node.inputs[0]
        while table_node.type == 'Identity':
            table_node = table_node.inputs[0]
        shared_name = table_node.get_attr_value("shared_name")
        utils.make_sure(shared_name is not None, "Could not determine table shared name for node %s", node.name)
        utils.make_sure(shared_name in initialized_tables, "Initialized table %s for node %s not found.",
                        shared_name, node.name)

        default_node = node.inputs[2]
        utils.make_sure(default_node.is_const(), "Default value of table lookup must be const.")
        default_val_np = default_node.get_tensor_value(as_list=False)
        default_val = default_node.get_tensor_value()

        dtype = ctx.get_dtype(node.output[0])
        in_dtype = ctx.get_dtype(node.input[1])
        #utils.make_sure(dtype == TensorProto.INT64 and in_dtype == TensorProto.STRING,
        #                "Only lookup tables of type string->int64 are currently supported.")

        #cats_strings, cats_int64s = initialized_tables[shared_name]
        shape = ctx.get_shape(node.output[0])

        node_name = node.name
        #node_inputs = node.input
        node_inputs = node.input.copy()
        node_outputs = node.output

        cats_strings, cats_int64s = initialized_tables[shared_name]
        if in_dtype != TensorProto.STRING:
            # cats_strings is a list of keys in bytes
            cats_strings = np.vectorize(lambda s: s.encode("UTF-8"))(cats_strings.astype(np.str))
            node_inputs[1] = ctx.make_node("Cast", [node_inputs[1]], attr={'to': TensorProto.STRING}).output[0]

        if node.inputs[1].is_const():
            # Handle explicitly since const folding doesn't work for tables
            #key_np = node.inputs[1].get_tensor_value(as_list=False)
            key_np = node.inputs[1].get_tensor_value(as_list=False).astype(np.str)
            ctx.remove_node(node.name)
            key_to_val = dict(zip(cats_strings, cats_int64s))
            def lookup_value(key):
                return key_to_val.get(key.encode("UTF-8"), default_val_np)
            lookup_result = np.vectorize(lookup_value)(key_np)
            onnx_tensor = numpy_helper.from_array(lookup_result, node_name)
            ctx.make_node("Const", name=node_name, inputs=[], outputs=node_outputs,
                          attr={"value": onnx_tensor}, shapes=[lookup_result.shape], dtypes=[dtype])
        else:
            ctx.remove_node(node.name)
            ctx.make_node("CategoryMapper", domain=constants.AI_ONNX_ML_DOMAIN,
                          name=node_name, inputs=[node_inputs[1]], outputs=node_outputs,
                          attr={'cats_int64s': cats_int64s, 'cats_strings': cats_strings, 'default_int64': default_val},
                          shapes=[shape], dtypes=[dtype])

        customer_nodes = ctx.find_output_consumers(table_node.output[0])
        if len(customer_nodes) == 0:
            ctx.remove_node(table_node.name)


@tf_op("LookupTableSizeV2")
class LookupTableSize:
    @classmethod
    def version_1(cls, ctx, node, initialized_tables, **kwargs):
        table_node = node.inputs[0]
        while table_node.type == 'Identity':
            table_node = table_node.inputs[0]
        shared_name = table_node.get_attr_value("shared_name")
        utils.make_sure(shared_name is not None, "Could not determine table shared name for node %s", node.name)
        utils.make_sure(shared_name in initialized_tables, "Initialized table %s for node %s not found.",
                        shared_name, node.name)
        keys, _ = initialized_tables[shared_name]

        node_name = node.name
        node_outputs = node.output
        ctx.remove_node(node.name)
        size_const = ctx.make_const(node_name, np.array(len(keys), dtype=np.int64))
        ctx.replace_all_inputs(node_outputs[0], size_const.output[0])

        customer_nodes = ctx.find_output_consumers(table_node.output[0])
        if len(customer_nodes) == 0:
            ctx.remove_node(table_node.name)
