import logging
from abc import ABC
from onnx import TensorProto
from tf2onnx.graph_matcher import GraphMatcher, OpTypePattern


logger = logging.getLogger(__name__)


class LowApiRewriterBase(ABC):
    def __init__(self, g):
        self.g = g
        self.scat_pat = OpTypePattern('ScatterElements', name='scatter_op')
        self.scat_pat = GraphMatcher(self.scat_pat, allow_reorder=False)

    def has_scatter_elements(self, ops):
        return len(list(self.scat_pat.match_ops(ops))) > 0


class LowApiBucketizeRewriter(LowApiRewriterBase):
    """
    Bucketize --> [ FloatBucketizePlugin | IntBucketizePlugin ]
    """
    def __init__(self, g):
        super().__init__(g)
        self.buck_pat = OpTypePattern('Bucketize', name='buck_op')
        self.buck_pat = GraphMatcher(self.buck_pat, allow_reorder=False)

    def rewrite(self, ops):
        matches = list(self.buck_pat.match_ops(ops))
        if matches:
            assert not self.has_scatter_elements(ops)

        success_float = 0
        success_int = 0
        for m in matches:
            op = m.get_op('buck_op')
            inp_op = op.inputs[0]

            # Forge new node.
            bounds = op.attr['boundaries']
            if inp_op.output_dtypes[0] == TensorProto.FLOAT:
                bounds = bounds.floats
                new_op = self.g.make_node(
                    'FloatBucketizePlugin',
                    inputs=inp_op.output.copy(),
                    attr={
                        'boundaries': bounds,
                        'boundaries_len': len(bounds),
                    },
                    dtypes=[TensorProto.INT64],
                    shapes=inp_op.output_shapes.copy(),
                )
                success_float += 1
            else:
                if len(bounds.ints) > 0:
                    bounds = bounds.ints
                elif len(bounds.floats) > 0:
                    bounds = bounds.floats
                    assert all(b.is_integer() for b in bounds)
                else:
                    raise NotImplementedError

                new_op = self.g.make_node(
                    'IntBucketizePlugin',
                    inputs=inp_op.output.copy(),
                    attr={
                        'boundaries': bounds,
                        'boundaries_len': len(bounds),
                    },
                    dtypes=[TensorProto.INT64],
                    shapes=inp_op.output_shapes.copy(),
                )
                success_int += 1

            # Rewire consumers to the new node.
            self.g.replace_all_inputs(op.output[0], new_op.output[0])
            self.g.remove_node(op.name)

        logger.info(f'Replaced {success_float} float and {success_int} integer Bucketize nodes.')
        return self.g.get_nodes()


def rewrite_low_api_bucketize(g, ops):
    return LowApiBucketizeRewriter(g).rewrite(ops)


class LowApiCategoryMapperRewriter(LowApiRewriterBase):
    """
    CategoryMapper --> CategoricalPlugin
    """
    def __init__(self, g):
        super().__init__(g)
        self.cm_pattern = OpTypePattern('CategoryMapper', name='cm_op', inputs=[
            OpTypePattern('Cast', name='cast_op')
        ])
        self.cm_pattern = GraphMatcher(self.cm_pattern, allow_reorder=False)

    def rewrite(self, ops):
        matches = list(self.cm_pattern.match_ops(ops))
        if matches:
            assert not self.has_scatter_elements(ops)

        for m in matches:
            inp_op = m.get_op('cast_op')
            cm_op = m.get_op('cm_op')

            vocab = list(map(int, cm_op.attr['cats_strings'].strings))
            new_op = self.g.make_node(
                'CategoricalPlugin',
                inputs=inp_op.input.copy(),
                attr={
                    'vocab_list': vocab,
                    'vocab_index': cm_op.attr['cats_int64s'].ints,
                    'vocab_len': len(vocab),
                    'default_value': cm_op.attr['default_int64'].i,
                    'is_feacol': False,
                },
                dtypes=[TensorProto.INT64],
                shapes=inp_op.output_shapes.copy(),
            )

            # Rewire consumers to the new node.
            self.g.replace_all_inputs(cm_op.output[0], new_op.output[0])
            self.g.remove_node(inp_op.name)
            self.g.remove_node(cm_op.name)

        logger.info(f'Replaced {len(matches)} categorical CategoryMapper nodes.')
        return self.g.get_nodes()


def rewrite_low_api_category_mapper(g, ops):
    return LowApiCategoryMapperRewriter(g).rewrite(ops)


class LowApiStringToHashBucketFastRewriter(LowApiRewriterBase):
    """
    AsString; StringToHashBucketFast --> StringToHashPlugin
    """
    def __init__(self, g):
        super().__init__(g)
        self.s2h_pattern = OpTypePattern('StringToHashBucketFast', name='s2h_op', inputs=[
            OpTypePattern('AsString', name='inp_op'),
        ])
        self.s2h_pattern = GraphMatcher(self.s2h_pattern, allow_reorder=False)

    def rewrite(self, ops):
        matches = list(self.s2h_pattern.match_ops(ops))
        if matches:
            assert not self.has_scatter_elements(ops)

        for m in matches:
            inp_op = m.get_op('inp_op')
            s2h_op = m.get_op('s2h_op')

            new_op = self.g.make_node(
                'StringToHashPlugin',
                inputs=inp_op.input.copy(),
                attr={
                    'num_buckets': s2h_op.attr['num_buckets'].i
                },
                dtypes=[TensorProto.INT64],
                shapes=inp_op.output_shapes.copy(),
            )

            # Rewire consumers to the new node.
            self.g.replace_all_inputs(s2h_op.output[0], new_op.output[0])
            self.g.remove_node(inp_op.name)
            self.g.remove_node(s2h_op.name)

        logger.info(f'Replaced {len(matches)} categorical StringToHashBucketFast nodes.')
        return self.g.get_nodes()


def rewrite_low_api_string_to_hash_bucket_fast(g, ops):
    return LowApiStringToHashBucketFastRewriter(g).rewrite(ops)
