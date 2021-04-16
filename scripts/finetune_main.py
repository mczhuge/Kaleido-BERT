# Copyright (c) 2021 Alibaba Group. Licensed under the MIT license.

import sys
import os
import tensorflow as tf
from easytransfer import base_model, FLAGS
from easytransfer import model_zoo
from easytransfer import layers
from easytransfer.losses import softmax_cross_entropy
from easytransfer import preprocessors
from easytransfer.datasets import BundleCSVReader
from easytransfer.losses import masked_language_model_loss, next_sentence_prediction_loss, image_reconstruction_kld_loss
from easytransfer.evaluators import masked_language_model_eval_metrics, next_sentence_prediction_eval_metrics
from kaleidobert_utils import prediction_analysis, PretrainConfig, append_to_file, delete_exists_file
from easytransfer.evaluators import classification_eval_metrics, matthew_corr_metrics

_app_flags = tf.app.flags
_app_flags.DEFINE_string("task", default=None, help='')
_app_flags.DEFINE_integer("input_sequence_length", default=None, help='')
_app_flags.DEFINE_integer("vocab_size", default=30522, help='')
_app_flags.DEFINE_integer("image_feature_size", default=None, help='')
_APP_FLAGS = _app_flags.FLAGS

class KaleidoBERTApplications(base_model):
    def __init__(self, **kwargs):
        super(KaleidoBERTApplications, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]
    
    def build_catepred_logits(self, input_features, mode=None):
        
        # 48/122 - total category/subcategory num
        dense = layers.Dense(48,
                             kernel_initializer=layers.get_initializer(0.01),
                             name='catepred_dense')

        if mode == tf.estimator.ModeKeys.TRAIN:
            input_features = tf.nn.dropout(input_features, keep_prob=0.9)
            #input_features = tf.nn.dropout(input_features, keep_prob=1.0)

        logits = dense(input_features)

        return logits
    
    def build_subcatepred_logits(self, input_features, mode=None):
        
        # 48/122 - total category/subcategory num
        dense = layers.Dense(122,
                             kernel_initializer=layers.get_initializer(0.01),
                             name='subcatepred_dense')

        if mode == tf.estimator.ModeKeys.TRAIN:
            input_features = tf.nn.dropout(input_features, keep_prob=0.9)
            #input_features = tf.nn.dropout(input_features, keep_prob=1.0)            


        logits = dense(input_features)

        return logits

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                      user_defined_config=self.user_defined_config)

        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)


        text_prod_id, input_ids, input_mask, segment_ids, prod_desc, image_prod_id, prod_img_id, \
            img_feature_convert_rotation, img_feature_convert_jigsaw, img_feature_convert_camouflage, \
            img_feature_convert_grey_mask, img_feature_convert_blank_mask, image_mask, \
            img_loc_position_rotation, img_loc_position_jigsaw, img_loc_position_camouflage, \
            img_loc_position_grey_mask, img_loc_position_blank_mask, img_category, \
            img_category_id, img_subcategory, img_subcategory_id = preprocessor(features)
        
        mlm_logits, nsp_logits, \
            rotation_logits, jigsaw_logits, camouflage_logits, \
            greymask_logits, target_raw_grey_features, \
            blankmask_logits, target_raw_blank_features, pooled_output = \
            model(input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    img_feature_convert_rotation=img_feature_convert_rotation,
                    img_feature_convert_jigsaw=img_feature_convert_jigsaw,
                    img_feature_convert_camouflage=img_feature_convert_camouflage,
                    img_feature_convert_greymask=img_feature_convert_grey_mask,
                    img_feature_convert_blankmask=img_feature_convert_blank_mask,
                    img_feature_rotation_position=img_loc_position_rotation,
                    img_feature_jigsaw_position=img_loc_position_jigsaw,
                    img_feature_camouflage_position=img_loc_position_camouflage,
                    img_feature_greymask_position=img_loc_position_grey_mask,
                    img_feature_blankmask_position=img_loc_position_blank_mask,
                    img_mask=image_mask,
                    output_features=False,
                    mode=mode)
        
        if _APP_FLAGS.task == "catepred":
            catePredLogits = self.build_catepred_logits(pooled_output, mode=mode)
            return catePredLogits, img_category_id
        elif _APP_FLAGS.task == "subcatepred":
            subCatePredLogits = self.build_subcatepred_logits(pooled_output, mode=mode)
            return subCatePredLogits, img_subcategory_id
        else:
            tf.logging.info("Kaleido-BERT fintune task {} is not support".format(_APP_FLAGS.task))
            return 

    def build_catepred_loss(self, logits, labels):
        return softmax_cross_entropy(labels, 48, logits)
    
    def build_subcatepred_loss(self, logits, labels):
        return softmax_cross_entropy(labels, 122, logits)

    def build_loss(self, logits, labels):
        if _APP_FLAGS.task == "catepred":
            return self.build_catepred_loss(logits, labels)
        elif _APP_FLAGS.task == "subcatepred":
            return self.build_subcatepred_loss(logits, labels)

    def build_catepred_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, 48)
    
    def build_subcatepred_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, 122)

    def build_eval_metrics(self, logits, labels):
        if _APP_FLAGS.task == "catepred":
            return self.build_catepred_eval_metrics(logits, labels)
        elif _APP_FLAGS.task == "subcatepred":
            return self.build_subcatepred_eval_metrics(logits, labels)

    def build_predictions(self, output):
        logits = output
        predictions = dict()
        predictions["logits"] = logits
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions

def main(_):
    config = PretrainConfig()
    app = KaleidoBERTApplications(user_defined_config=config)

    if FLAGS.mode == "train_and_evaluate":
        train_reader = BundleCSVReader(input_glob=app.train_input_fp,
                                       is_training=True,
                                       shuffle_buffer_size=4096,
                                       input_schema=app.input_schema,
                                       batch_size=app.train_batch_size,
                                       worker_hosts=app.config.worker_hosts,
                                       task_index=app.config.task_index
                                       )
        eval_reader = BundleCSVReader(input_glob=app.eval_input_fp,
                                      input_schema=app.input_schema,
                                      is_training=False,
                                      batch_size=app.eval_batch_size,
                                      worker_hosts=app.config.worker_hosts,
                                      task_index=app.config.task_index)

        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

if __name__ == "__main__":
    tf.app.run()

