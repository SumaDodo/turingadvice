import os
import sys
import re
from absl import flags

import tensorflow.compat.v1 as tf

from reward.comparative.model import ComparativeRewardModel
from reward.comparative.data import SEQUENCE_LENGTH, MODEL_DIR

def _define_flags():
    flags.DEFINE_string(
        name="input_path",
        default=None,
        help="Path to a tab-separated text file with columns [inputs, targets]"
    )
    flags.DEFINE_string(
        name="reward_file_path",
        default=None,
        help="Path to reward model predictions file path"
    )
    flags.DEFINE_string(
        name="N",
        default=None,
        help="Number of targets for best of N"
    )
    return flags.FLAGS

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    OUTPUT_FILE = os.path.join(FLAGS.tmp_dir, "Reward_output.txt")
    max_reward_index = []

    #from the reward model output file get the predictions with highest reward value
    with tf.io.gfile.GFile(FLAGS.reward_file_path,"r") as rewards:
        block = [line for line in rewards]
        block = [line.strip() for line in block[:FLAGS.N]]
        cur_pos = 0
        while block:
            index = cur_pos * FLAGS.N + my_block.index(max(my_block))
            max_reward_index.append(index)
            cur_pos += 1
            my_block = [line.strip() for line in block[cur_pos * FLAGS.N:(cur_pos + 1) * FLAGS.N]]

    #from N predictions for each input pick the one with highest reward value
    with tf.io.gfile.GFile(FLAGS.input_path, "r") as pred, tf.io.gfile.GFile(OUTPUT_FILE, "w") as out:
        texts = [re.sub(r'\s+»\s+', '\n\n', text).strip() for text in pred.read().splitlines()]
        for i in max_reward_index:
            out.write(texts[i]+"\n")


if __name__ == "__main__":
    tf.app.run()