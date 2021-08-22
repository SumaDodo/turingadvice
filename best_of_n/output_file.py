import os
import sys
from absl import flags
import tensorflow as tf

def _define_flags():
    flags.DEFINE_string(
        name="tmp_dir",
        default=None,
        help="Temporary dir for internal use of BestOfNGenerator"
    )
    return flags.FLAGS

def main(_):
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    questions = []
    FINAL_OUTPUT_FILE = os.path.join(FLAGS.tmp_dir, "Output.txt")
    TMP_REPEATS_PATH = os.path.join(FLAGS.tmp_dir, "BoN-repeats.txt")
    OUTPUT_PRED_FILE = os.path.join(FLAGS.tmp_dir, "output_test.tsv-1037496")

    with tf.io.gfile.GFile(TMP_REPEATS_PATH, "r") as quest:
        questions = quest['selftext']

    with tf.io.gfile.GFile(FINAL_OUTPUT_FILE,"w") as output_file, tf.io.gfile.GFile(OUTPUT_PRED_FILE, "r") as target:
        for q, t in zip(questions, target):
            output_file.write(q + "\t" + t + "\n")

if __name__ == "__main__":
    tf.app.run()