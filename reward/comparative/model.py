import os
from reward.comparative.data.ops import SEQUENCE_LENGTH
from tqdm import tqdm
from copy import deepcopy

import gin
import tensorflow.compat.v1 as tf
import mesh_tensorflow
from mesh_tensorflow.transformer import utils

from reward.comparative.data import get_dataset, get_checkpoint_paths
from reward.comparative.mtf_extensions import\
  make_reward_bitransformer, _tpu_estimator_model_fn
from t5.data import get_mixture_or_task
from t5.models.mtf_model import \
  MtfModel, _get_latest_checkpoint_from_dir, _operative_config_path
from reward.comparative.data import \
  get_dataset, get_prediction_dataset, get_checkpoint_paths
from reward.comparative.mtf_extensions import \
  make_reward_bitransformer, _tpu_estimator_model_fn, _predict_reward_fn

REDDIT_TASK_NAME = "reddit_v002"

class ComparativeRewardModel(MtfModel):
  def __init__(self, *args, **kwargs):
    # Monkey-patch Mesh-Tensorflow model instantiation
    mesh_tensorflow.transformer.transformer.make_bitransformer = \
        make_reward_bitransformer
    # Monkey-patch Mesh-Tensorflow TPUEstimator creation
    mesh_tensorflow.transformer.utils.tpu_estimator_model_fn = \
        _tpu_estimator_model_fn
    super(ComparativeRewardModel, self).__init__(*args, **kwargs)
    self._predict_fn = _predict_reward_fn

  def train(self, bucket_name, dataset_id, steps, init_checkpoint=None):
    """
    This method is a combination of MtfModel.train and
    mtf.transformer.utils.train_model, which MtfModel.train calls. It was
    re-written to fit our tfrecords dataset, which is already tokenized.

    Args:
    dataset_id: str
      Dataset id. See reward/comparative/ops.py
    steps: int
      Number of training steps.
    init_checkpoint: str
      Read from this checkpoint path when initializing variables.
    """
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    sequence_length = deepcopy(self._sequence_length)
    sequence_length.update({"targets": sequence_length["targets"] * 2})
    estimator = self.estimator(vocabulary, init_checkpoint, sequence_length)
    def input_fn(params):
      del params
      dataset = get_dataset(
        bucket_name=bucket_name,
        dataset_id=dataset_id,
        split="train",
        from_local=False,
        from_tfrecords=False,
        stack_answer_pairs=True,
        shuffle_buffer_size=1000
      )
      dataset = dataset.repeat().batch(
          self.batch_size * (self._ensemble_inputs or 1),
          drop_remainder=True
        )
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset
    estimator.train(input_fn=input_fn, max_steps=steps)

  def eval(
    self, bucket_name, dataset_id, split="val", min_checkpoint_steps=None,
    tokens_per_microbatch_per_replica=None
    ):
    """
    Evaluate model metrics on several checkpoints
    """
    ckpt_paths = get_checkpoint_paths(self._model_dir, min_checkpoint_steps)
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    sequence_length = deepcopy(self._sequence_length)
    sequence_length.update({"targets": sequence_length["targets"] * 2})
    # "I have no idea why but I think this must be needed?" - Rowan
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
      gin.bind_parameter(
        "serialize_num_microbatches.tokens_per_microbatch_per_replica",
        tokens_per_microbatch_per_replica
      )
    estimator = self.estimator(vocabulary, sequence_length=sequence_length)
    # 
    # Data input function for TPUEstimator
    def _input_fn(params):
      del params
      dataset = get_dataset(
        bucket_name=bucket_name,
        dataset_id=dataset_id,
        split=split,
        from_local=False,
        from_tfrecords=False,
        stack_answer_pairs=True,
        shuffle_buffer_size=-1
      )
      dataset = dataset.batch(self.batch_size, drop_remainder=True)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset
    # Count steps in dataset
    with tf.Session() as sess:
      dataset = _input_fn(None)
      steps_in_dataset = sess.run(dataset.reduce(0, lambda x,_: x + 1))
      steps_in_dataset = int(steps_in_dataset)
    # Evaluate all checkpoints beyond min_checkpoint_steps
    for ckpt_path in tqdm(ckpt_paths):
      metrics = estimator.evaluate(
        input_fn=_input_fn,
        steps=steps_in_dataset,
        checkpoint_path=ckpt_path,
        name=split
      )
      print(f"Metrics for ckpt '{ckpt_path}': {metrics}")

  def finetune(
    self, bucket_name, dataset_id, finetune_steps, pretrained_model_dir,
    dropout_rate=0.1, tokens_per_microbatch_per_replica=None,
    pretrained_checkpoint_step=-1
    ):
    if pretrained_checkpoint_step == -1:
      checkpoint_step = _get_latest_checkpoint_from_dir(pretrained_model_dir)
    else:
      checkpoint_step = pretrained_checkpoint_step
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(pretrained_model_dir))
      gin.bind_parameter("%dropout_rate", dropout_rate)
      gin.bind_parameter(
        "serialize_num_microbatches.tokens_per_microbatch_per_replica",
        tokens_per_microbatch_per_replica
      )
      # gin.bind_parameter("tpu_estimator_model_fn.tpu_summaries", True)
    model_ckpt = "model.ckpt-" + str(checkpoint_step)
    self.train(
      bucket_name=bucket_name,
      dataset_id=dataset_id,
      steps=checkpoint_step + finetune_steps,
      init_checkpoint=os.path.join(pretrained_model_dir, model_ckpt)
    )
  
  def predict_from_file(self, input_path, output_path, checkpoint_steps=-1):
    """
    Args:
    input_file: str
      Path to a tab-separated text file with columns [inputs, targets]
    """
    if checkpoint_steps == -1:
      checkpoint_steps = _get_latest_checkpoint_from_dir(self._model_dir)
    with gin.unlock_config():
      gin.parse_config_file(_operative_config_path(self._model_dir))
    vocabulary = get_mixture_or_task(REDDIT_TASK_NAME).get_vocabulary()
    estimator = self.estimator(vocabulary, sequence_length=SEQUENCE_LENGTH)
    def _input_fn(params):
      del params
      dataset = get_prediction_dataset(input_path, batch_size=self.batch_size)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset
    predictions_iter = estimator.predict(
      input_fn=_input_fn,
      checkpoint_path=f"{self._model_dir}/model.ckpt-{checkpoint_steps}"
    )
    if tf.io.gfile.exists(output_path):
      tf.io.gfile.remove(output_path)
    with tf.io.gfile.GFile(output_path, "w") as output_file:
      for predictions in predictions_iter:
        if isinstance(predictions, list):
          for prediction in predictions:
            output_file.write(str(prediction["outputs"]) + "\n")
        else:
          output_file.write(str(predictions["outputs"]) + "\n")

  def estimator(self, vocabulary, init_checkpoint=None, sequence_length=None):
    """
    A version of MtfModel.estimator which also accepts the `sequence_length`
    parameter.
    """
    return utils.get_estimator(
        model_type=self._model_type,
        input_vocab_size=utils.inputs_vocabulary(vocabulary).vocab_size,
        output_vocab_size=utils.targets_vocabulary(vocabulary).vocab_size,
        layout_rules=self._layout_rules,
        mesh_shape=self._mesh_shape,
        model_dir=self._model_dir,
        batch_size=self.batch_size,
        sequence_length=sequence_length or self._sequence_length,
        autostack=self._autostack,
        learning_rate_schedule=self._learning_rate_schedule,
        keep_checkpoint_max=self._keep_checkpoint_max,
        save_checkpoints_steps=self._save_checkpoints_steps,
        optimizer=self._optimizer,
        predict_fn=self._predict_fn,
        variable_filter=self._variable_filter,
        ensemble_inputs=self._ensemble_inputs,
        use_tpu=self._tpu,
        tpu_job_name=self._tpu_job_name,
        iterations_per_loop=self._iterations_per_loop,
        cluster=self._cluster,
        init_checkpoint=init_checkpoint)
