# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

from easytransfer import layers
from .modeling_utils import PretrainedConfig, PreTrainedModel


KALEIDOBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'pai-kaleidobert-base-en': "kaleidobert/pai-kaleidobert-base-en/model.ckpt"
}

KALEIDOBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'pai-kaleidobert-base-en': "kaleidobert/pai-kaleidobert-base-en/config.json"
}

class KaleidoBERTConfig(PretrainedConfig):
    """Configuration for `KaleidoBERT`.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      patch_feature_size: patch feature size
      max_patch_position_embeddings: max_patch_position_embeddings

    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 max_position_embeddings,
                 type_vocab_size,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 patch_type_vocab_size=2,
                 patch_feature_size=2048,
                 max_patch_position_embeddings=64,
                 **kwargs):
        super(KaleidoBERTConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.patch_type_vocab_size = patch_type_vocab_size
        self.patch_feature_size = patch_feature_size
        self.max_patch_position_embeddings = max_patch_position_embeddings

class RotationEmbedding(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(RotationEmbedding, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.img_rotation_seq_len = config.img_rotation_seq_len

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "rotation_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        # TODO: Shared by other image features
        self.patch_type_embeddings = self.add_weight(
            "rotation_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(RotationEmbedding, self).build(input_shape)

    def call(self, inputs, position_dense, training=False):
        img_feature_convert_rotation, img_feature_rotation_segment, img_feature_rotation_position = inputs
        
        input_rotation_feature = self.dropout_input(img_feature_convert_rotation, training=training)
        rotation_embeddings = tf.einsum("abc,cd->abd",
                                     input_rotation_feature, self.image_projector)

        input_shape = layers.get_shape_list(rotation_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(img_feature_rotation_segment, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        img_feature_rotation_position = tf.reshape(tf.cast(img_feature_rotation_position, tf.float32), [batch_size, self.img_rotation_seq_len, 5])
        position_embeddings = tf.einsum("abc,cd->abd", img_feature_rotation_position, position_dense)
        position_embeddings = tf.reshape(position_embeddings, [batch_size, seq_length, width])

        embeddings = rotation_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="rotationEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings

class JigsawEmbedding(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(JigsawEmbedding, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.img_jigsaw_seq_len = config.img_jigsaw_seq_len

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "jigsaw_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        # TODO: Shared by other image features
        self.patch_type_embeddings = self.add_weight(
            "jigsaw_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(JigsawEmbedding, self).build(input_shape)

    def call(self, inputs, position_dense, training=False):
        img_feature_convert_jigsaw, img_feature_jigsaw_segment, img_feature_jigsaw_position = inputs

        input_jigsaw_feature = self.dropout_input(img_feature_convert_jigsaw, training=training)
        jigsaw_embeddings = tf.einsum("abc,cd->abd",
                                     input_jigsaw_feature, self.image_projector)

        input_shape = layers.get_shape_list(jigsaw_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(img_feature_jigsaw_segment, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        img_feature_jigsaw_position = tf.reshape(tf.cast(img_feature_jigsaw_position, tf.float32), [batch_size, self.img_jigsaw_seq_len, 5])
        position_embeddings = tf.einsum("abc,cd->abd", img_feature_jigsaw_position, position_dense)
        position_embeddings = tf.reshape(position_embeddings, [batch_size, seq_length, width])

        embeddings = jigsaw_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="jigsawEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings

class CamouflageEmbedding(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(CamouflageEmbedding, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.img_camouflage_seq_len = config.img_camouflage_seq_len

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "camouflage_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        # TODO: Shared by other image features
        self.patch_type_embeddings = self.add_weight(
            "camouflage_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(CamouflageEmbedding, self).build(input_shape)

    def call(self, inputs, position_dense, training=False):
        img_feature_convert_camouflage, img_feature_camouflage_segment, img_feature_camouflage_position = inputs

        input_camouflage_feature = self.dropout_input(img_feature_convert_camouflage, training=training)
        camouflage_embeddings = tf.einsum("abc,cd->abd",
                                     input_camouflage_feature, self.image_projector)

        input_shape = layers.get_shape_list(camouflage_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(img_feature_camouflage_segment, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        img_feature_camouflage_position = tf.reshape(tf.cast(img_feature_camouflage_position, tf.float32), [batch_size, self.img_camouflage_seq_len, 5])
        position_embeddings = tf.einsum("abc,cd->abd", img_feature_camouflage_position, position_dense)
        position_embeddings = tf.reshape(position_embeddings, [batch_size, seq_length, width])

        embeddings = camouflage_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="camouflageEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings

class GreymaskEmbedding(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(GreymaskEmbedding, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.img_greymask_seq_len = config.img_greymask_seq_len

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "greymask_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        # TODO: Shared by other image features
        self.patch_type_embeddings = self.add_weight(
            "greymask_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(GreymaskEmbedding, self).build(input_shape)

    def call(self, inputs, position_dense, training=False):
        img_feature_convert_greymask, img_feature_greymask_segment, img_feature_greymask_position = inputs

        input_greymask_feature = self.dropout_input(img_feature_convert_greymask, training=training)
        greymask_embeddings = tf.einsum("abc,cd->abd",
                                     input_greymask_feature, self.image_projector)

        input_shape = layers.get_shape_list(greymask_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(img_feature_greymask_segment, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        img_feature_greymask_position = tf.reshape(tf.cast(img_feature_greymask_position, tf.float32), [batch_size, self.img_greymask_seq_len, 5])
        position_embeddings = tf.einsum("abc,cd->abd", img_feature_greymask_position, position_dense)
        position_embeddings = tf.reshape(position_embeddings, [batch_size, seq_length, width])

        embeddings = greymask_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="greymaskEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings

class BlankmaskEmbedding(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(BlankmaskEmbedding, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.img_blankmask_seq_len = config.img_blankmask_seq_len

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "blankmask_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        # TODO: Shared by other image features
        self.patch_type_embeddings = self.add_weight(
            "blankmask_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(BlankmaskEmbedding, self).build(input_shape)

    def call(self, inputs, position_dense, training=False):
        img_feature_convert_blankmask, img_feature_blankmask_segment, img_feature_blankmask_position = inputs

        input_blankmask_feature = self.dropout_input(img_feature_convert_blankmask, training=training)
        blankmask_embeddings = tf.einsum("abc,cd->abd",
                                     input_blankmask_feature, self.image_projector)

        input_shape = layers.get_shape_list(blankmask_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(img_feature_blankmask_segment, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        img_feature_blankmask_position = tf.reshape(tf.cast(img_feature_blankmask_position, tf.float32), [batch_size, self.img_blankmask_seq_len, 5])
        position_embeddings = tf.einsum("abc,cd->abd", img_feature_blankmask_position, position_dense)
        position_embeddings = tf.reshape(position_embeddings, [batch_size, seq_length, width])

        embeddings = blankmask_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="blankmaskEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings

# Rotation Task
class KaleidoBERTRotationHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(KaleidoBERTRotationHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.config = config
        self.rotation_label_num = config.img_rotation_label_num
        self.rotation_seq_len = config.img_rotation_seq_len
    
    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.rotation_label_num, self.hidden_size * self.rotation_seq_len],
                                    initializer=layers.get_initializer(self.config.initializer_range),
                                    trainable=True, name="rotation_output_weights")
        self.bias = self.add_weight(shape=(self.rotation_label_num,),
                                    initializer="zeros", trainable=True, name="rotation_output_bias")
        
        super(KaleidoBERTRotationHead, self).build(input_shape)
    
    def call(self, rotation_seq_output):
        rotation_seq_output = tf.reshape(rotation_seq_output, [-1, self.hidden_size * self.rotation_seq_len])
        logits = tf.matmul(rotation_seq_output, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

class KaleidoBERTJigsawHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(KaleidoBERTJigsawHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.config = config
        self.jigsaw_label_num = config.img_jigsaw_label_num
        self.jigsaw_seq_len   = config.img_jigsaw_seq_len
    
    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.jigsaw_label_num, self.hidden_size * self.jigsaw_seq_len],
                                    initializer=layers.get_initializer(self.config.initializer_range),
                                    trainable=True, name="jiasaw_output_weights")
        self.bias = self.add_weight(shape=(self.jigsaw_label_num,),
                                    initializer="zeros", trainable=True, name="jigsaw_output_bias")
        
        super(KaleidoBERTJigsawHead, self).build(input_shape)
    
    def call(self, jigsaw_seq_output):
        jigsaw_seq_output = tf.reshape(jigsaw_seq_output, [-1, self.hidden_size * self.jigsaw_seq_len])
        logits = tf.matmul(jigsaw_seq_output, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

class KaleidoBERTCamouflageHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(KaleidoBERTCamouflageHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.config = config
        self.camouflage_label_num = config.img_camouflage_label_num
        self.camouflage_seq_len   = config.img_camouflage_seq_len
    
    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.camouflage_label_num, self.hidden_size * self.camouflage_seq_len],
                                    initializer=layers.get_initializer(self.config.initializer_range),
                                    trainable=True, name="camouflage_output_weights")
        self.bias = self.add_weight(shape=(self.camouflage_label_num,),
                                    initializer="zeros", trainable=True, name="camouflage_output_bias")
        
        super(KaleidoBERTCamouflageHead, self).build(input_shape)
    
    def call(self, camouflage_seq_output):
        camouflage_seq_output = tf.reshape(camouflage_seq_output, [-1, self.hidden_size * self.camouflage_seq_len])
        logits = tf.matmul(camouflage_seq_output, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)

        return logits

# Greymask Task: equal to Masked Patch Modeling
class KaleidoBERTGreymaskHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(KaleidoBERTGreymaskHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.patch_feature_size = config.patch_feature_size
        self.initializer_range = config.initializer_range
        
    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.hidden_size, self.patch_feature_size],
                                              initializer=layers.get_initializer(self.initializer_range),
                                              trainable=True, name="greymask_output_weights")

        super(KaleidoBERTGreymaskHead, self).build(input_shape)

    def call(self, image_seq_output, masked_patch_positions):
        pred_patch_features = layers.gather_indexes(image_seq_output, masked_patch_positions)
        logits = tf.matmul(pred_patch_features, self.output_weights)
        return logits

# Blankmask Task: equal to Masked Patch Modeling
class KaleidoBERTBlankmaskHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(KaleidoBERTBlankmaskHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.patch_feature_size = config.patch_feature_size
        self.initializer_range = config.initializer_range

    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.hidden_size, self.patch_feature_size],
                                              initializer=layers.get_initializer(self.initializer_range),
                                              trainable=True, name="blanmask_output_weights")

        super(KaleidoBERTBlankmaskHead, self).build(input_shape)

    def call(self, image_seq_output, masked_patch_positions):
        pred_patch_features = layers.gather_indexes(image_seq_output, masked_patch_positions)
        logits = tf.matmul(pred_patch_features, self.output_weights)
        return logits


class KaleidoBERTBackbone(layers.Layer):

    def __init__(self, config, **kwargs):
        super(KaleidoBERTBackbone, self).__init__(**kwargs)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = layers.BertEmbeddings(config, name="embeddings")
        
        self.rotation_embedding = RotationEmbedding(config, name="rotation_embeddings")
        self.jigsaw_embedding   = JigsawEmbedding(config, name="jigsaw_embeddings")
        self.camouflage_embedding = CamouflageEmbedding(config, name="camouflage_embeddings")
        self.greymask_embedding = GreymaskEmbedding(config, name="greymask_embeddings")
        self.blankmask_embedding    = BlankmaskEmbedding(config, name="blank_embeddings")

        self.encoder = layers.Encoder(config, name="encoder")
        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense")

    def build(self, input_shape):
        # shared position dense network
        self.position_dense = self.add_weight(
            "position_dense",
            dtype=tf.float32,
            shape=[5, self.config.hidden_size],
            initializer=layers.get_initializer(self.config.initializer_range)
        )

    def call(self, inputs, training=False):
        input_ids, input_mask, segment_ids, \
            img_feature_convert_rotation, img_feature_rotation_mask, img_feature_rotation_segment, img_feature_rotation_position, \
            img_feature_convert_jigsaw, img_feature_jigsaw_mask, img_feature_jigsaw_segment, img_feature_jigsaw_position, \
            img_feature_convert_camouflage, img_feature_camouflage_mask, img_feature_camouflage_segment, img_feature_camouflage_position, \
            img_feature_convert_greymask, img_feature_greymask_mask, img_feature_greymask_segment, img_feature_greymask_position, \
            img_feature_convert_blankmask, img_feature_blankmask_mask, img_feature_blankmask_segment, img_feature_blankmask_position \
            = inputs
    
        # step 1: Process text feature
        token_embedding_output = self.embeddings([input_ids, segment_ids], training=training)

        # step 2: Process rotation feature
        rotation_embedding_inputs = [img_feature_convert_rotation, img_feature_rotation_segment, img_feature_rotation_position] 
        rotation_embedding_output = self.rotation_embedding(rotation_embedding_inputs, self.position_dense, training=training)

        # step 3: Process jigsaw feature 
        jigsaw_embedding_inputs = [img_feature_convert_jigsaw, img_feature_jigsaw_segment, img_feature_jigsaw_position]
        jigsaw_embedding_output = self.jigsaw_embedding(jigsaw_embedding_inputs, self.position_dense, training=training)

        # step 4: Process camouflage feature 
        camouflage_embedding_inputs = [img_feature_convert_camouflage, img_feature_camouflage_segment, img_feature_camouflage_position]
        camouflage_embedding_output = self.camouflage_embedding(camouflage_embedding_inputs, self.position_dense, training=training)

        # step 5: Process greymask feature 
        greymask_embedding_inputs = [img_feature_convert_greymask, img_feature_greymask_segment, img_feature_greymask_position]
        greymask_embedding_output = self.greymask_embedding(greymask_embedding_inputs, self.position_dense, training=training)

        # step 6: blankmask feature
        blankmask_embedding_inputs = [img_feature_convert_blankmask, img_feature_blankmask_segment, img_feature_blankmask_position]
        blankmask_embedding_output = self.blankmask_embedding(blankmask_embedding_inputs, self.position_dense, training=training)

        # step 7: concate
        embedding_output = tf.concat([token_embedding_output, rotation_embedding_output, \
                                        jigsaw_embedding_output, camouflage_embedding_output, \
                                        greymask_embedding_output, blankmask_embedding_output ], \
                                        axis=1)

        # step 8: attention_mask
        attention_mask = layers.get_attn_mask_kaleidobert(input_ids, input_mask, \
                                img_feature_convert_rotation, img_feature_rotation_mask, \
                                img_feature_convert_jigsaw, img_feature_jigsaw_mask, \
                                img_feature_convert_camouflage, img_feature_camouflage_mask, \
                                img_feature_convert_greymask, img_feature_greymask_mask, \
                                img_feature_convert_blankmask, img_feature_blankmask_mask)

        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs


class KaleidoBERTPreTrainedModel(PreTrainedModel):
    config_class = KaleidoBERTConfig
    pretrained_model_archive_map = KALEIDOBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = KALEIDOBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(KaleidoBERTPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = KaleidoBERTBackbone(config, name="bert")
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")
        self.rotation_head = KaleidoBERTRotationHead(config, name="cls/img_rotation")
        self.jigsaw_head   = KaleidoBERTJigsawHead(config, name="cls/img_jigsaw")
        self.camouflage_head = KaleidoBERTCamouflageHead(config, name="cls/img_camouflage")
        self.greymask_head   = KaleidoBERTGreymaskHead(config, name="cls/img_greymask")
        self.blankmask_head  = KaleidoBERTBlankmaskHead(config, name="cls/img_blankmask")

    def blankmask_patch_features(self, patch_features, masked_patch_positions):
        onehot_image_mask = tf.reduce_sum(tf.one_hot(masked_patch_positions, self.config.img_blankmask_seq_len, dtype=tf.float32),
                                          axis=1)
        reverse_onehot_image_mask = 1 - (onehot_image_mask[:, :, tf.newaxis])
        masked_patch = tf.multiply(patch_features, reverse_onehot_image_mask)
        return masked_patch

    def call(self, 
             input_ids=None,
             input_mask=None,
             segment_ids=None,
             masked_lm_positions=None,
             img_feature_convert_rotation=None,
             img_feature_convert_jigsaw=None,
             img_feature_convert_camouflage=None,
             img_feature_convert_greymask=None,
             img_feature_convert_blankmask=None,
             img_feature_rotation_position=None,
             img_feature_jigsaw_position=None,
             img_feature_camouflage_position=None,
             img_feature_greymask_position=None,
             img_feature_blankmask_position=None,
             img_mask=None,
             img_feature_rotation_mask=None,
             img_feature_jigsaw_mask=None,
             img_feature_camouflage_mask=None,
             img_feature_greymask_mask=None,
             img_feature_blankmask_mask=None,
             img_feature_rotation_segment=None,
             img_feature_jigsaw_segment=None,
             img_feature_camouflage_segment=None,
             img_feature_greymask_segment=None,
             img_feature_blankmask_segment=None,
             img_subtasks_flag=None,
             img_grey_lm_ids=None,
             img_blank_lm_ids=None,
             **kwargs):

        """
        Examples::

            model = model_zoo.get_pretrained_model('icbu-fashionbert-small-en')

            mlm_logits, nsp_logits, mpm_logits, target_raw_patch_features = \
                model(input_ids,
                      input_mask=input_mask,
                      segment_ids=token_type_ids,
                      image_feature=image_feature,
                      image_mask=image_mask,
                      masked_lm_positions=lm_positions,
                      masked_patch_positions=masked_patch_positions,
                      output_features=False,
                      mode=mode)

        """

        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN
        
        batch_size = 0
        if input_ids is not None:
            input_shape = layers.get_shape_list(input_ids)
            batch_size = input_shape[0]
        else: 
            input_shape = layers.get_shape_list(img_feature_convert_rotation)
            batch_size = input_shape[0]

        # Step 1: Prepare text sequence input
        text_seq_length = 0
        if input_ids is not None:
            input_shape = layers.get_shape_list(input_ids)
            batch_size = input_shape[0]
            text_seq_length = input_shape[1]
            if input_mask is None:
                input_mask = tf.ones(shape=[batch_size, text_seq_length], dtype=tf.int32)
            if segment_ids is None:
                segment_ids = tf.zeros(shape=[batch_size, text_seq_length], dtype=tf.int32)
            if masked_lm_positions is None:
                masked_lm_positions = tf.ones(shape=[batch_size, text_seq_length], dtype=tf.int64)
        else:
            text_seq_length = self.config.text_seq_len
            input_ids = tf.zeros(shape=[batch_size, self.config.text_seq_len], dtype=tf.int32)
            if input_mask is None:
                input_mask = tf.zeros(shape=[batch_size, self.config.text_seq_len], dtype=tf.int32)
            if segment_ids is None:
                segment_ids = tf.zeros(shape=[batch_size, self.config.text_seq_len], dtype=tf.int32)
            if masked_lm_positions is None:
                masked_lm_positions = tf.zeros(shape=[batch_size, self.config.text_seq_len], dtype=tf.int64)

        # Step 2: Prepare image rotation sequence input
        img_feature_rotation_end_position = text_seq_length + self.config.img_rotation_seq_len
        if img_feature_convert_rotation is not None:
            img_feature_convert_rotation = tf.reshape(img_feature_convert_rotation, 
                                            [batch_size, self.config.img_rotation_seq_len, self.config.patch_feature_size])
            if img_feature_rotation_mask is None:
                img_feature_rotation_mask = tf.ones(shape=[batch_size, self.config.img_rotation_seq_len], dtype=tf.int32)
            if img_feature_rotation_segment is None:
                img_feature_rotation_segment = 1 * tf.ones(shape=[batch_size, self.config.img_rotation_seq_len], dtype=tf.int32)
            if img_feature_rotation_position is None:
                img_feature_rotation_position = tf.ones(shape=[batch_size, self.config.img_rotation_seq_len * 5], dtype=tf.int32)
        else:
            img_feature_convert_rotation = tf.zeros(shape=[batch_size, self.config.img_rotation_seq_len, \
                self.config.patch_feature_size], dtype=tf.float32)
            if img_feature_rotation_mask is None:
                img_feature_rotation_mask = tf.zeros(shape=[batch_size, self.config.img_rotation_seq_len], dtype=tf.int32)
            if img_feature_rotation_segment is None:
                img_feature_rotation_segment = 1 * tf.ones(shape=[batch_size, self.config.img_rotation_seq_len], dtype=tf.int32)
            if img_feature_rotation_position is None:
                img_feature_rotation_position = tf.zeros(shape=[batch_size, self.config.img_rotation_seq_len * 5], dtype=tf.int32)

        # Step 3: Prepare image jigsaw sequence input 
        img_feature_jigsaw_end_position = img_feature_rotation_end_position + self.config.img_jigsaw_seq_len
        if img_feature_convert_jigsaw is not None:
            img_feature_convert_jigsaw =  tf.reshape(img_feature_convert_jigsaw, 
                                            [batch_size, self.config.img_jigsaw_seq_len, self.config.patch_feature_size])  
            if img_feature_jigsaw_mask is None:
                img_feature_jigsaw_mask = tf.ones(shape=[batch_size, self.config.img_jigsaw_seq_len], dtype=tf.int32)
            if img_feature_jigsaw_segment is None:
                img_feature_jigsaw_segment = 2 * tf.ones(shape=[batch_size, self.config.img_jigsaw_seq_len], dtype=tf.int32)
            if img_feature_jigsaw_position is None:
                img_feature_jigsaw_position = tf.ones(shape=[batch_size, self.config.img_jigsaw_seq_len * 5], dtype=tf.int32)
        else:
            img_feature_convert_jigsaw =  tf.zeros(shape=[batch_size, self.config.img_jigsaw_seq_len, \
                self.config.patch_feature_size], dtype=tf.float32)  
            if img_feature_jigsaw_mask is None:
                img_feature_jigsaw_mask = tf.zeros(shape=[batch_size, self.config.img_jigsaw_seq_len], dtype=tf.int32)
            if img_feature_jigsaw_segment is None:
                img_feature_jigsaw_segment = 2 * tf.ones(shape=[batch_size, self.config.img_jigsaw_seq_len], dtype=tf.int32)
            if img_feature_jigsaw_position is None:
                img_feature_jigsaw_position = tf.zeros(shape=[batch_size, self.config.img_jigsaw_seq_len * 5], dtype=tf.int32)
            
        
        # Step 4: Prepare image camouflage sequence input 
        img_feature_camouflage_end_position = img_feature_jigsaw_end_position + self.config.img_camouflage_seq_len
        if img_feature_convert_camouflage is not None:
            img_feature_convert_camouflage = tf.reshape(img_feature_convert_camouflage, 
                                            [batch_size, self.config.img_camouflage_seq_len, self.config.patch_feature_size])  
            if img_feature_camouflage_mask is None:
                img_feature_camouflage_mask = tf.ones(shape=[batch_size, self.config.img_camouflage_seq_len], dtype=tf.int32)
            if img_feature_camouflage_segment is None:
                img_feature_camouflage_segment = 3 * tf.ones(shape=[batch_size, self.config.img_camouflage_seq_len], dtype=tf.int32)
            if img_feature_camouflage_position is None:
                img_feature_camouflage_position = tf.ones(shape=[batch_size, self.config.img_camouflage_seq_len * 5], dtype=tf.int32)
        else:
            img_feature_convert_camouflage = tf.zeros(shape=[batch_size, self.config.img_camouflage_seq_len, \
                self.config.patch_feature_size], dtype=tf.float32)  
            if img_feature_camouflage_mask is None:
                img_feature_camouflage_mask = tf.zeros(shape=[batch_size, self.config.img_camouflage_seq_len], dtype=tf.int32)
            if img_feature_camouflage_segment is None:
                img_feature_camouflage_segment = 3 * tf.ones(shape=[batch_size, self.config.img_camouflage_seq_len], dtype=tf.int32)
            if img_feature_camouflage_position is None:
                img_feature_camouflage_position = tf.zeros(shape=[batch_size, self.config.img_camouflage_seq_len * 5], dtype=tf.int32)

        # step 5: Prepare image greymask sequence input 
        img_feature_greymask_end_position = img_feature_camouflage_end_position + self.config.img_greymask_seq_len
        if img_feature_convert_greymask is not None:
            img_feature_convert_greymask = tf.reshape(img_feature_convert_greymask, 
                                            [batch_size, self.config.img_greymask_seq_len, self.config.patch_feature_size])   
            if img_feature_greymask_mask is None:
                img_feature_greymask_mask = tf.ones(shape=[batch_size, self.config.img_greymask_seq_len], dtype=tf.int32)
            if img_feature_greymask_segment is None:
                img_feature_greymask_segment = 4 * tf.ones(shape=[batch_size, self.config.img_greymask_seq_len], dtype=tf.int32)
            if img_feature_greymask_position is None:
                img_feature_greymask_position = tf.ones(shape=[batch_size, self.config.img_greymask_seq_len * 5], dtype=tf.int32)
        else:
            img_feature_convert_greymask = tf.zeros(shape=[batch_size, self.config.img_greymask_seq_len, \
                self.config.patch_feature_size], dtype=tf.float32)   
            if img_feature_greymask_mask is None:
                img_feature_greymask_mask = tf.zeros(shape=[batch_size, self.config.img_greymask_seq_len], dtype=tf.int32)
            if img_feature_greymask_segment is None:
                img_feature_greymask_segment = 4 * tf.ones(shape=[batch_size, self.config.img_greymask_seq_len], dtype=tf.int32)
            if img_feature_greymask_position is None:
                img_feature_greymask_position = tf.zeros(shape=[batch_size, self.config.img_greymask_seq_len * 5], dtype=tf.int32)

        # step 6: Prepare image blankmask sequence input 
        img_feature_blankmask_end_position = img_feature_greymask_end_position + self.config.img_blankmask_seq_len
        if img_feature_convert_blankmask is not None:
            img_feature_convert_blankmask = tf.reshape(img_feature_convert_blankmask, 
                                            [batch_size, self.config.img_blankmask_seq_len, self.config.patch_feature_size])  
            if img_feature_blankmask_mask is None:
                img_feature_blankmask_mask = tf.ones(shape=[batch_size, self.config.img_blankmask_seq_len], dtype=tf.int32)
            if img_feature_blankmask_segment is None:
                img_feature_blankmask_segment = 5 * tf.ones(shape=[batch_size, self.config.img_blankmask_seq_len], dtype=tf.int32)
            if img_feature_blankmask_position is None:
                img_feature_blankmask_position = tf.ones(shape=[batch_size, self.config.img_blankmask_seq_len * 5], dtype=tf.int32)
        else:
            img_feature_convert_blankmask = tf.zeros(shape=[batch_size, self.config.img_blankmask_seq_len, \
                self.config.patch_feature_size], dtype=tf.float32)  
            if img_feature_blankmask_mask is None:
                img_feature_blankmask_mask = tf.zeros(shape=[batch_size, self.config.img_blankmask_seq_len], dtype=tf.int32)
            if img_feature_blankmask_segment is None:
                img_feature_blankmask_segment = 5 * tf.ones(shape=[batch_size, self.config.img_blankmask_seq_len], dtype=tf.int32)
            if img_feature_blankmask_position is None:
                img_feature_blankmask_position = tf.zeros(shape=[batch_size, self.config.img_blankmask_seq_len * 5], dtype=tf.int32)

        # TODO: TO DETERMINE whether replace the mask patches in input_sequences
        if kwargs['mode'] == tf.estimator.ModeKeys.PREDICT:
            masked_img_feature_convert_blankmask = img_feature_convert_blankmask
            masked_img_feature_convert_greymask = img_feature_convert_greymask
        else:
            if img_blank_lm_ids is not None:
                masked_img_feature_convert_blankmask = self.blankmask_patch_features(img_feature_convert_blankmask, img_blank_lm_ids)
            else:
                masked_img_feature_convert_blankmask = img_feature_convert_blankmask
            
            masked_img_feature_convert_greymask = img_feature_convert_greymask

        if img_grey_lm_ids is None:
            img_grey_lm_ids = tf.ones(shape=[batch_size, self.config.masked_greymask_patch_num], dtype=tf.int64)

        if img_blank_lm_ids is None:
            img_blank_lm_ids = tf.ones(shape=[batch_size, self.config.masked_blankmask_patch_num], dtype=tf.int64)

        inputs = [  input_ids, input_mask, segment_ids, \
                    img_feature_convert_rotation, img_feature_rotation_mask, img_feature_rotation_segment, img_feature_rotation_position, \
                    img_feature_convert_jigsaw, img_feature_jigsaw_mask, img_feature_jigsaw_segment, img_feature_jigsaw_position, \
                    img_feature_convert_camouflage, img_feature_camouflage_mask, img_feature_camouflage_segment, img_feature_camouflage_position, \
                    masked_img_feature_convert_greymask, img_feature_greymask_mask, img_feature_greymask_segment, img_feature_greymask_position, \
                    masked_img_feature_convert_blankmask, img_feature_blankmask_mask, img_feature_blankmask_segment, img_feature_blankmask_position, \
                    ]
        

        if kwargs.get("output_features", True) == True:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            return sequence_output, pooled_output
        else:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            
            text_sequence_output       = sequence_output[:, :text_seq_length, :]
            rotation_sequence_output   = sequence_output[:, text_seq_length:img_feature_rotation_end_position, :]
            jigsaw_sequence_output     = sequence_output[:, img_feature_rotation_end_position:img_feature_jigsaw_end_position, :]
            camouflage_sequence_output = sequence_output[:, img_feature_jigsaw_end_position:img_feature_camouflage_end_position, :]
            greymask_sequence_output   = sequence_output[:, img_feature_camouflage_end_position:img_feature_greymask_end_position, :]
            blankmask_sequence_output  = sequence_output[:, img_feature_greymask_end_position:img_feature_blankmask_end_position, :]

            mlm_logits = self.mlm(text_sequence_output, masked_lm_positions)
            nsp_logits = self.nsp(pooled_output)
            rotation_logits  = self.rotation_head(rotation_sequence_output)
            jigsaw_logits    = self.jigsaw_head(jigsaw_sequence_output)
            camouflage_logits= self.camouflage_head(camouflage_sequence_output)
            greymask_logits  = self.greymask_head(greymask_sequence_output, img_grey_lm_ids)
            blankmask_logits = self.blankmask_head(blankmask_sequence_output, img_blank_lm_ids)

            target_raw_grey_features = layers.gather_indexes(img_feature_convert_greymask, img_grey_lm_ids)
            target_raw_blank_features = layers.gather_indexes(img_feature_convert_blankmask, img_blank_lm_ids)

            return mlm_logits, nsp_logits, \
                   rotation_logits, jigsaw_logits, camouflage_logits, \
                   greymask_logits, target_raw_grey_features, \
                   blankmask_logits, target_raw_blank_features, pooled_output

