import logging
import numpy as np
from functools import reduce
import operator
from onnx import TensorProto
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.graph import get_upstream_nodes


logger = logging.getLogger(__name__)


def rewrite_feature_column(g, ops):
    root = OpTypePattern('Placeholder', name='placeholder_node')
    root = GraphMatcher(root, allow_reorder=False)

    block_start = OpTypePattern('Bucketize|CategoryMapper|StringToHashBucketFast', name='first_node')
    block_start = GraphMatcher(block_start, allow_reorder=False)

    gather_op = OpTypePattern('Gather', name='gather_op', inputs=[
        OpTypePattern('Const', name='gather_input'),
        '*'
    ])
    gather_op = GraphMatcher(gather_op, allow_reorder=False)

    scatter_op = OpTypePattern('ScatterElements', name='scatter_op')
    scatter_op = GraphMatcher(scatter_op, allow_reorder=False)

    block_end = [
        # Reshape end style.
        OpTypePattern('*', name='next_node', inputs=[
            OpTypePattern('Reshape', name='last_node', inputs=[
                OpTypePattern('Reshape'),
                OpTypePattern('Cast', inputs=[
                    OpTypePattern('Concat', inputs=[
                        OpTypePattern('Unsqueeze', inputs=[OpTypePattern('Squeeze', inputs=['Slice'])]),
                        OpTypePattern('*'),
                    ]),
                ]),
            ]),
        ]),
        # Add end style.
        OpTypePattern('Add', name='last_node', inputs=[
            OpTypePattern('Mul', inputs=[
                OpTypePattern('Cast', inputs=['Tile']),
                OpTypePattern('Cast', inputs=[
                    OpTypePattern('Mul', inputs=[
                        OpTypePattern('Cast', inputs=['Reshape']),
                        'Const',
                    ]),
                ]),
            ]),
            OpTypePattern('Mul', inputs=[
                OpTypePattern('Reshape', inputs=['Div', 'Concat']),
                OpTypePattern('Cast', inputs=[
                    OpTypePattern('Not', inputs=['Tile']),
                ]),
            ])
        ]),
    ]
    block_end = [GraphMatcher(block_end, allow_reorder=False) for block_end in block_end]

    success = 0
    success_2 = 0
    matches = []
    for j, m in enumerate(block_end):
        if len(matches) > 0:
            break
        else:
            logger.info(f'Trying end-block pattern #{j}...')

        for i, e in enumerate(m.match_ops(ops)):
            print(i, end=', ', flush=True)
            # x = e.get_op('next_node')
            e = e.get_op('last_node')
            # if '1127_' not in e.name:
            #    continue
            up_nodes = get_upstream_nodes([e])
            up_nodes = [n for seq in up_nodes for n in seq]

            s = [m.get_op('first_node') for m in block_start.match_ops(up_nodes)]
            if len(s) > 1:
                logger.warning('Mapper association is ambiguous. I just match the first upstream one.')
            elif len(s) == 0:
                logger.warning(f'{e.name}: could not find a Bucketize|CategoryMapper|StringToHashBucketFast.')
                continue
            s = s[0]

            t = [m.get_op('gather_op') for m in gather_op.match_ops(up_nodes)]
            if len(t) > 1:
                logger.warning(
                    'Search for gather operation was ambiguous. Will just take the first one. This could be wrong.'
                )
            t = t[0]

            r = [m.get_op('placeholder_node') for m in root.match_ops(up_nodes)]
            if len(r) > 1:
                logger.warning('Search for root node was ambiguous. Will just take the first one. This could be wrong.')
            r = r[0]

            # If we can also find a scatter node, we have feature column.
            is_feacol = len([m.get_op('scatter_op') for m in scatter_op.match_ops(up_nodes)]) > 0

            matches.append((r, s, e, t, is_feacol, up_nodes))
            success += 1
            success_2 += (1 if is_feacol else 0)
    print('')

    logger.info(f'Successfully identified {success} / {success_2} feature column constructs!')

    for idx, (r, s, e, t, is_feacol, up_nodes) in enumerate(matches):
        # if False:
        #     print(f'[{i}] {s.op.op_type}, ', end='')
        # elif False:
        #     print(
        #         f'{idx}:\n',
        #         'r', r.op.op_type, '\t', r.name, '\n',
        #         's', s.op.op_type, '\t', s.name, '\n',
        #         'e', e.op.op_type, '\t', e.name, '\n',
        #         'g', g.op.op_type, '\t', g.name
        #     )

        # Replace subtree with with EmbeddingPlugin.
        i = t.inputs[0]
        val = i.get_tensor_value(as_list=False)
        i.set_tensor_value(np.ravel(val))
        i.attr['value'].name = 'word_embeddings'
        g.remove_node(e.name)
        emb_node = g.make_node(
            'EmbeddingPlugin',
            inputs=r.output.copy(),
            attr={
                'word_embeddings': i.attr['value'],
                'embedding_hidden_size': val.shape[-1],
                'embedding_weight_length': reduce(operator.mul, val.shape, 1),
            },
            outputs=e.output.copy(),
            dtypes=[TensorProto.FLOAT] * len(e.output),
            shapes=[[-1, val.shape[-1]]] * len(e.output)
        )

        # Insert feature column datatype specific decoding plugins.
        if s.op.op_type == 'Bucketize':
            bounds = s.attr['boundaries']
            if r.output_dtypes[0] == TensorProto.FLOAT:
                bounds = bounds.floats
                pre_node = g.make_node(
                    'FloatBucketizePlugin',
                    inputs=r.output.copy(),
                    attr={
                        'boundaries': bounds,
                        'boundaries_len': len(bounds),
                    },
                    dtypes=[TensorProto.INT64],
                    shapes=[[-1, val.shape[-1]]],
                )
            else:
                if len(bounds.ints) > 0:
                    bounds = bounds.ints
                elif len(bounds.floats) > 0:
                    bounds = bounds.floats
                    assert all(b.is_integer() for b in bounds)
                else:
                    raise NotImplementedError

                pre_node = g.make_node(
                    'IntBucketizePlugin',
                    inputs=r.output.copy(),
                    attr={
                        'boundaries': bounds,
                        'boundaries_len': len(bounds),
                    },
                    dtypes=[TensorProto.INT64],
                    shapes=[[-1, val.shape[-1]]],
                )
        elif s.op.op_type == 'CategoryMapper':
            vocab = list(map(int, s.attr['cats_strings'].strings))
            pre_node = g.make_node(
                'CategoricalPlugin',
                inputs=r.output.copy(),
                attr={
                    'vocab_list': vocab,
                    'vocab_index': s.attr['cats_int64s'].ints,
                    'vocab_len': len(vocab),
                    'default_value': s.attr['default_int64'].i,
                    'is_feacol': is_feacol,
                },
                dtypes=[TensorProto.INT64],
                shapes=[[-1, val.shape[-1]]],
            )
        elif s.op.op_type == 'StringToHashBucketFast':
            pre_node = g.make_node(
                'StringToHashPlugin',
                inputs=r.output.copy(),
                attr={
                    'num_buckets': s.attr['num_buckets'].i
                },
                dtypes=[TensorProto.INT64],
                shapes=[[-1, val.shape[-1]]],
            )
        else:
            logger.warning(
                f'{idx}:\n',
                'r', r.op.op_type, '\t', r.name, '\n',
                's', s.op.op_type, '\t', s.name, '\n',
                'e', e.op.op_type, '\t', e.name, '\n',
                'g', g.op.op_type, '\t', g.name
            )
            logger.warning('Not a known pattern...')
            pre_node = None

        if pre_node:
            g.replace_inputs(emb_node, pre_node.output)
            g.remove_node(s.name)

    return g.get_nodes()
