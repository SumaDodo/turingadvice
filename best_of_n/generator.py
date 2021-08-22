import os

import tensorflow as tf
from mesh_tensorflow.transformer import utils

class BestOfNGenerator():
    def __init__(
        self, t5_model, t5_model_ckpt_steps, N, sampling_keep_top_p, tmp_dir
        ):
        self.t5_model = t5_model
        self.model_ckpt_steps = t5_model_ckpt_steps
        self.N = N
        self.sampling_keep_top_p = sampling_keep_top_p
        self.tmp_dir = tmp_dir
    
    def generate_N(self, inputs_path, outputs_path):
        # Repeat each input N times, store in temporary file
        TMP_REPEATS_PATH = os.path.join(self.tmp_dir, "BoN-repeats.txt")
        with tf.io.gfile.GFile(inputs_path, "r") as inputs_file, \
             tf.io.gfile.GFile(TMP_REPEATS_PATH, "w") as repeats_file:
            for line in inputs_file:
                for _ in range(self.N):
                    repeats_file.write(line + "\n")
        # Predict over repeated inputs file
        self.t5_model.predict(
            input_file=TMP_REPEATS_PATH,
            output_file=outputs_path,
            checkpoint_steps=self.model_ckpt_steps,
            sampling_keep_top_p=self.sampling_keep_top_p
        )

        FINAL_OUTPUT_FILE = os.path.join(self.tmp_dir, "Output.txt")
        with tf.io.gfile.GFile(TMP_REPEATS_PATH, "r") as quest, tf.io.gfile.GFile(FINAL_OUTPUT_FILE, "w") as output_file, tf.io.gfile.GFile(outputs_path, "r") as target:
            for q, t in zip(quest, target):
                FINAL_OUTPUT_FILE.write(q+"\t"+t)
