# SPDX-License-Identifier: Apache-2.0

"""tf2onnx.rewriter module."""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tf2onnx.rewriter.cond_rewriter import rewrite_cond
from tf2onnx.rewriter.conv2d_with_pad_rewriter import rewrite_conv2d_with_pad
from tf2onnx.rewriter.dropout_rewriter import rewrite_dropout
from tf2onnx.rewriter.eye_rewriter import rewrite_eye
from tf2onnx.rewriter.flatten_rewriter import rewrite_flatten
from tf2onnx.rewriter.gemm_rewriter import rewrite_gemm
from tf2onnx.rewriter.leakyrelu_rewriter import rewrite_leakyrelu
from tf2onnx.rewriter.random_normal_rewriter import rewrite_random_normal
from tf2onnx.rewriter.random_uniform import rewrite_random_uniform, rewrite_random_uniform_fold_const
from tf2onnx.rewriter.rnn import rewrite_single_direction_lstm, rewrite_bi_direction_lstm, \
    rewrite_single_direction_gru, rewrite_bi_direction_gru, \
    rewrite_custom_rnn_cell, rewrite_generic_loop
from tf2onnx.rewriter.thresholded_relu_rewriter import rewrite_thresholded_relu
from tf2onnx.rewriter.transpose_rewriter import rewrite_transpose
from tf2onnx.rewriter.conv2d_with_add_rewriter import rewrite_biasadd_with_conv2d
from tf2onnx.rewriter.quantization_ops_rewriter import rewrite_quantize_and_dequantize
from tf2onnx.rewriter.layer_normalization_rewriter import rewrite_layer_normalization
from tf2onnx.rewriter.trt_input_name_rewriter import rewrite_input_name_for_trt
from tf2onnx.rewriter.low_api_rewriters import \
    rewrite_low_api_bucketize, rewrite_low_api_category_mapper, rewrite_low_api_string_to_hash_bucket_fast
from tf2onnx.rewriter.string_to_hash_plugin_rewriter import rewrite_string_to_hash_plugin
from tf2onnx.rewriter.feature_column_rewriter import rewrite_feature_column

__all__ = [
    "rewrite_cond",
    "rewrite_conv2d_with_pad",
    "rewrite_dropout",
    "rewrite_eye",
    "rewrite_flatten",
    "rewrite_gemm",
    "rewrite_leakyrelu",
    "rewrite_random_normal",
    "rewrite_random_uniform",
    "rewrite_random_uniform_fold_const",
    "rewrite_thresholded_relu",
    "rewrite_transpose",
    "rewrite_single_direction_lstm",
    "rewrite_bi_direction_lstm",
    "rewrite_single_direction_gru",
    "rewrite_bi_direction_gru",
    "rewrite_custom_rnn_cell",
    "rewrite_generic_loop",
    "rewrite_biasadd_with_conv2d",
    "rewrite_quantize_and_dequantize",
    "rewrite_layer_normalization",
    # BOSS Zhipin additions.
    'rewrite_input_name_for_trt',
    'rewrite_low_api_bucketize',
    'rewrite_low_api_category_mapper',
    'rewrite_low_api_string_to_hash_bucket_fast',
    'rewrite_string_to_hash_plugin',
    'rewrite_feature_column',
]
