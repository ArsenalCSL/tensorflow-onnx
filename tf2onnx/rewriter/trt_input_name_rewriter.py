# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewrite - cut :0 from placeholders to make them TensorRT compatible.
"""
import re


def rewrite_input_name_for_trt(g, ops):
    old_inputs = g.inputs.copy()

    # Craft new inputs with the name we want.
    for old_input in old_inputs:
        o = old_input.output[0]
        n = o.rsplit(':', 1)[0]
        if o != n:
            dt = g.get_dtype(o)
            sh = g.get_shape(o)
            g.add_graph_input(n, dt, sh)

    # Rewire nodes to the new inputs and erase the old input.
    for old_input in old_inputs:
        o = old_input.output[0]
        n = o.rsplit(':', 1)[0]
        if o != n:
            g.replace_all_inputs(o, n)
            g.remove_node(old_input.name)

    return g.get_nodes()
