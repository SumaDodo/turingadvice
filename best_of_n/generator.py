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
        tmp_file_name = "BoN-repeats"+str(self.N)+".txt"
        TMP_REPEATS_PATH = os.path.join(self.tmp_dir,tmp_file_name)
        with tf.io.gfile.GFile(inputs_path, "r") as inputs_file, \
             tf.io.gfile.GFile(TMP_REPEATS_PATH, "w") as repeats_file:
            for line in inputs_file:
                for _ in range(self.N):
                    repeats_file.write(line)
        # Predict over repeated inputs file
        self.t5_model.predict(
            input_file=TMP_REPEATS_PATH,
            output_file=outputs_path,
            checkpoint_steps=self.model_ckpt_steps,
            sampling_keep_top_p=self.sampling_keep_top_p
        )
