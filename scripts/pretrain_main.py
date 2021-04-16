# Copyright (c) 2021 Alibaba Group. Licensed under the MIT license.

import tensorflow as tf
import numpy as np
import time
import os, sys
from easytransfer import base_model, FLAGS
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import BundleCSVReader
from easytransfer.losses import masked_language_model_loss, next_sentence_prediction_loss, image_reconstruction_kld_loss, image_reconstruction_kld_loss_with_mask
from easytransfer.evaluators import masked_language_model_eval_metrics, next_sentence_prediction_eval_metrics
from kaleidobert_utils import prediction_analysis, PretrainConfig, append_to_file, delete_exists_file

_app_flags = tf.app.flags
_app_flags.DEFINE_string("type", default=None, help='')
_app_flags.DEFINE_integer("input_sequence_length", default=None, help='')
_app_flags.DEFINE_integer("vocab_size", default=30522, help='')
_app_flags.DEFINE_integer("image_feature_size", default=None, help='')
_APP_FLAGS = _app_flags.FLAGS

def kaleidobert_eval_rotation_metrics(rotation_logits, rotation_labels):
    rotation_log_probs = tf.nn.log_softmax(rotation_logits, axis=-1)
    rotation_log_probs = tf.reshape(
        rotation_log_probs, [-1, rotation_log_probs.shape[-1]])
    rotation_predictions = tf.argmax(
        rotation_log_probs, axis=-1, output_type=tf.int32)
    rotation_labels = tf.reshape(rotation_labels, [-1])
    rotation_accuracy = tf.metrics.accuracy(
        labels=rotation_labels, predictions=rotation_predictions)

    metric_dict = {
        "rotation_accuracy": rotation_accuracy
    }

    return metric_dict

def kaleidobert_eval_jigsaw_metrics(jigsaw_logits, jigsaw_labels):
    jigsaw_log_probs = tf.nn.log_softmax(jigsaw_logits, axis=-1)
    jigsaw_log_probs = tf.reshape(
        jigsaw_log_probs, [-1, jigsaw_log_probs.shape[-1]])
    jigsaw_predictions = tf.argmax(
        jigsaw_log_probs, axis=-1, output_type=tf.int32)
    jigsaw_labels = tf.reshape(jigsaw_labels, [-1])
    jigsaw_accuracy = tf.metrics.accuracy(
        labels=jigsaw_labels, predictions=jigsaw_predictions)

    metric_dict = {
        "jigsaw_accuracy": jigsaw_accuracy
    }

    return metric_dict

def kaleidobert_eval_camouflage_metrics(camouflage_logits, camouflage_labels):
    camouflage_log_probs = tf.nn.log_softmax(camouflage_logits, axis=-1)
    camouflage_log_probs = tf.reshape(
        camouflage_log_probs, [-1, camouflage_log_probs.shape[-1]])
    camouflage_predictions = tf.argmax(
        camouflage_log_probs, axis=-1, output_type=tf.int32)
    camouflage_labels = tf.reshape(camouflage_labels, [-1])
    camouflage_accuracy = tf.metrics.accuracy(
        labels=camouflage_labels, predictions=camouflage_predictions)

    metric_dict = {
        "camouflage_accuracy": camouflage_accuracy
    }

    return metric_dict

def kaleidobert_rotation_loss(rotation_logits, img_rotation_gt, label_size, img_subtasks_flag = None):
    log_probs = tf.nn.log_softmax(rotation_logits, axis=-1)
    img_rotation_gt = tf.reshape(img_rotation_gt, [-1])
    one_hot_labels = tf.one_hot(img_rotation_gt, depth=label_size, dtype=tf.float32)
    if img_subtasks_flag is not None:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs * img_subtasks_flag, axis=-1)
    else:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    kaleidobert_rotation_loss = tf.reduce_mean(per_example_loss)
    tf.summary.scalar("rotation_loss", kaleidobert_rotation_loss)

    return kaleidobert_rotation_loss

def kaleidobert_jigsaw_loss(jigsaw_logits, img_jigsaw_gt, label_size, img_subtasks_flag = None):
    log_probs = tf.nn.log_softmax(jigsaw_logits, axis=-1)
    img_jigsaw_gt = tf.reshape(img_jigsaw_gt, [-1])
    one_hot_labels = tf.one_hot(img_jigsaw_gt, depth=label_size, dtype=tf.float32)
    if img_subtasks_flag is not None:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs * img_subtasks_flag, axis=-1)
    else:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    kaleidobert_jigsaw_loss = tf.reduce_mean(per_example_loss)
    tf.summary.scalar("jigsaw_loss", kaleidobert_jigsaw_loss)
    return kaleidobert_jigsaw_loss

def kaleidobert_camouflage_loss(camouflage_logits, img_camouflage_gt, label_size, img_subtasks_flag = None):
    log_probs = tf.nn.log_softmax(camouflage_logits, axis=-1)
    img_camouflage_gt = tf.reshape(img_camouflage_gt, [-1])
    one_hot_labels = tf.one_hot(img_camouflage_gt, depth=label_size, dtype=tf.float32)
    if img_subtasks_flag is not None:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs * img_subtasks_flag, axis=-1)
    else:
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    kaleidobert_camouflage_loss = tf.reduce_mean(per_example_loss)
    return kaleidobert_camouflage_loss 

class KaleidoBERTPretrain(base_model):

    def __init__(self, **kwargs):
        super(KaleidoBERTPretrain, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]
        tf.logging.info("Kaleido-BERT: user_defined_config {} ".format(self.user_defined_config))

    def build_logits(self, features, mode=None):

        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                           app_model_name="pretrain_language_model",
                                                           feature_type="pretrain_kaleidobert",
                                                           user_defined_config=self.user_defined_config)

        self.model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path,
                                                    input_sequence_length=_APP_FLAGS.input_sequence_length)

        if mode == tf.estimator.ModeKeys.PREDICT:
            text_prod_id, input_ids, input_mask, segment_ids, prod_desc, nx_sent_labels, \
                image_prod_id, prod_img_id, img_feature_convert_rotation, img_feature_convert_jigsaw, \
                img_feature_convert_camouflage, img_feature_convert_grey_mask, img_feature_convert_blank_mask, \
                image_mask, img_loc_position_rotation, img_loc_position_jigsaw, img_loc_position_camouflage, \
                img_loc_position_grey_mask, img_loc_position_blank_mask = preprocessor(features)

            mlm_logits, nsp_logits, \
                rotation_logits, jigsaw_logits, camouflage_logits, \
                greymask_logits, target_raw_grey_features, \
                blankmask_logits, target_raw_blank_features, pooled_output = \
                self.model(input_ids,
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

            logits = (mlm_logits, nsp_logits, rotation_logits, jigsaw_logits, \
                        camouflage_logits, greymask_logits, blankmask_logits)
            labels = (nx_sent_labels, target_raw_grey_features, target_raw_blank_features)

            return logits, labels
            
        else:
            input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, \
            masked_lm_weights, img_feature_convert_rotation, img_feature_convert_jigsaw, img_feature_convert_camouflage, \
            img_feature_convert_grey_mask,img_feature_convert_blank_mask,image_mask,img_loc_position_rotation, \
            img_loc_position_jigsaw,img_loc_position_camouflage,img_loc_position_grey_mask, \
            img_loc_position_blank_mask,img_subtasks_flag,img_grey_lm_ids,img_blank_lm_ids,img_rotation_gt, \
            img_jigsaw_gt, img_camouflage_gt, img_grey_mask_gt, nx_sent_labels = preprocessor(features)

            mlm_logits, nsp_logits, \
                rotation_logits, jigsaw_logits, camouflage_logits, \
                greymask_logits, target_raw_grey_features, \
                blankmask_logits, target_raw_blank_features, pooled_output = \
                self.model(input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        masked_lm_positions=masked_lm_positions,
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
                        img_grey_lm_ids=img_grey_lm_ids,
                        img_blank_lm_ids=img_blank_lm_ids,
                        output_features=False,
                        mode=mode)

            logits = (mlm_logits, nsp_logits, rotation_logits, jigsaw_logits, \
                        camouflage_logits, greymask_logits, blankmask_logits)
            labels = (masked_lm_ids, masked_lm_weights, nx_sent_labels, \
                        target_raw_grey_features, target_raw_blank_features, \
                        img_subtasks_flag,img_grey_lm_ids,img_blank_lm_ids,img_rotation_gt, \
                        img_jigsaw_gt, img_camouflage_gt, img_grey_mask_gt)

            return logits, labels

    def build_loss(self, logits, labels):
        mlm_logits, nsp_logits, rotation_logits, jigsaw_logits, \
            camouflage_logits, greymask_logits, blankmask_logits = logits
        masked_lm_ids, masked_lm_weights, nx_sent_labels, \
            target_raw_grey_features, target_raw_blank_features, \
            img_subtasks_flag,img_grey_lm_ids,img_blank_lm_ids,img_rotation_gt, \
            img_jigsaw_gt, img_camouflage_gt, img_grey_mask_gt = labels
        
        img_subtasks_flag = tf.cast(img_subtasks_flag, tf.float32)

        masked_lm_loss = masked_language_model_loss(mlm_logits, masked_lm_ids, masked_lm_weights,
                                                    _APP_FLAGS.vocab_size)
        next_sentence_loss = next_sentence_prediction_loss(nsp_logits, nx_sent_labels)

        rotation_loss = kaleidobert_rotation_loss(rotation_logits, img_rotation_gt, self.model.config.img_rotation_label_num)

        jigsaw_loss = kaleidobert_jigsaw_loss(jigsaw_logits, img_jigsaw_gt, self.model.config.img_jigsaw_label_num)

        camouflage_loss = kaleidobert_camouflage_loss(camouflage_logits, img_camouflage_gt, self.model.config.img_camouflage_label_num)

        greymask_loss = image_reconstruction_kld_loss_with_mask(greymask_logits, img_grey_mask_gt,
                                    self.model.config.masked_greymask_patch_num,
                                    self.model.config.patch_feature_size, 
                                    name="greymask_kld_loss")

        blankmask_loss = image_reconstruction_kld_loss_with_mask(blankmask_logits, target_raw_blank_features,
                                    self.model.config.masked_blankmask_patch_num,
                                    self.model.config.patch_feature_size, 
                                    name="blankmask_kld_loss")
        
        adaptive_loss = masked_lm_loss + next_sentence_loss + rotation_loss + jigsaw_loss + \
                        camouflage_loss + greymask_loss + blankmask_loss  
        return adaptive_loss

    def build_eval_metrics(self, logits, labels):
        mlm_logits, nsp_logits, rotation_logits, jigsaw_logits, \
            camouflage_logits, greymask_logits, blankmask_logits = logits
        masked_lm_ids, masked_lm_weights, nx_sent_labels, \
            target_raw_grey_features, target_raw_blank_features, \
            img_subtasks_flag,img_grey_lm_ids,img_blank_lm_ids,img_rotation_gt, \
            img_jigsaw_gt, img_camouflage_gt, img_grey_mask_gt = labels

        metrics = {}
        mlm_metrics = masked_language_model_eval_metrics(mlm_logits, masked_lm_ids, masked_lm_weights,
                                                         self.model.config.vocab_size) 
        metrics.update(mlm_metrics)

        nsp_metrics = next_sentence_prediction_eval_metrics(nsp_logits, nx_sent_labels)
        metrics.update(nsp_metrics)

        rotation_metrics = kaleidobert_eval_rotation_metrics(rotation_logits, img_rotation_gt)
        metrics.update(rotation_metrics)

        jigsaw_metrics = kaleidobert_eval_jigsaw_metrics(jigsaw_logits, img_jigsaw_gt)
        metrics.update(jigsaw_metrics)

        camouflage_metrics = kaleidobert_eval_camouflage_metrics(camouflage_logits, img_camouflage_gt)
        metrics.update(camouflage_metrics)

        return metrics

    def build_predictions(self, output):
        logits, _ = output
        mlm_logits, nsp_logits, rotation_logits, jigsaw_logits, \
            camouflage_logits, greymask_logits, blankmask_logits = logits
        return {"nsp_logits": nsp_logits}

def main():

    config = PretrainConfig()
    app = KaleidoBERTPretrain(user_defined_config=config)

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

    elif FLAGS.mode == "predict":

        predict_reader = BundleCSVReader(input_glob=app.predict_input_fp,
                                      input_schema=app.input_schema,
                                      batch_size=app.predict_batch_size,
                                      worker_hosts=app.config.worker_hosts,
                                      task_index=app.config.task_index)


        localtime = time.strftime("%Y%m%d-%H%M-", time.localtime())

        if _APP_FLAGS.type == "img2txt":
           print("============== predict task img2txt ==============")
           result_filename = "eval_img2txt_results.txt"
           analysis_type = "img2txt"
        else:
           print("============== predict task txt2img ==============")
           result_filename = "eval_txt2img_results.txt"
           analysis_type = "txt2img"

        if not tf.gfile.Exists(_APP_FLAGS.output_dir):
            tf.gfile.MkDir(_APP_FLAGS.output_dir)

        result_fp_path = os.path.join(_APP_FLAGS.output_dir, str(localtime) +  result_filename)
        print("result_fp_path: ", result_fp_path)
        delete_exists_file(result_fp_path)
        for result in app.run_predict(reader=predict_reader,
                                      checkpoint_path=app.config.predict_checkpoint_path,
                                      yield_single_examples=False):
            nsp_logits = result["nsp_logits"]
            labels = result["nx_sent_labels"]
            text_prod_id = result["text_prod_id"]
            image_prod_id = result["image_prod_id"]
            prod_img_id = result["prod_img_id"]

            batch_pred_result = str(text_prod_id.tolist()) + "\001" \
                                + str(image_prod_id.tolist()) + "\001" \
                                + str(prod_img_id.tolist()) + "\001" \
                                + str(labels.tolist()) + "\001" \
                                + str(np.reshape(nsp_logits, [-1]).tolist()) + "\n"
            
            append_to_file(result_fp_path, batch_pred_result)
        prediction_analysis(result_fp_path, type=analysis_type)

if __name__ == "__main__":
    main()


