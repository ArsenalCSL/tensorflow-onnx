import numpy as np
import tf2onnx.utils as utils
from tf2onnx.graph_matcher import GraphMatcher, OpTypePattern


def rewrite_string_to_hash_plugin(g, ops):
    s2h_pattern = OpTypePattern('StringToHashPlugin', name='s2h_op')
    s2h_pattern = GraphMatcher(s2h_pattern, allow_reorder=False)

    matches = list(s2h_pattern.match_ops(ops))
    for m in matches:
        s2h_op = m.get_op('s2h_op')
        if s2h_op.attr['num_buckets'].i != 1:
            continue

        # Create a zero constant and replace the s2h with it.
        zero_const = g.make_const(utils.make_name("zero_const"), np.array(0, dtype=np.int64))
        for out in s2h_op.output:
            g.replace_all_inputs(out, zero_const.output[0])

        # Drop the old value.
        g.remove_node(s2h_op.name)

    return g.get_nodes()
