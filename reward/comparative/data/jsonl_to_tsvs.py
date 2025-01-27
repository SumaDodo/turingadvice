import os
import sys
import json
from time import time
from tqdm import tqdm
from absl import flags
from random import choice
from datetime import datetime
from contextlib import ExitStack

from data.assertions import question_is_valid, answer_is_valid, answer_pair_is_valid
from data.to_tfrecord_t5 import encoder, _trim_to_desired_length, _fix_reddit_text
from reward.comparative.data import SELFTEXT_DESIRED_LEN, LOCAL_TSV_PATH, SPLITS

META_OUT_PATH = os.path.join(os.path.dirname(LOCAL_TSV_PATH), "meta.json")

def _define_flags():
    flags.DEFINE_string(
        name="dataset_id",
        default=None,
        help="Id of the resulting dataset. Will use timestamp if unspecified"
    )
    flags.DEFINE_string(
        name="jsonl_path",
        default="data/redditadvice2019.jsonl",
        help="Dataset generated by create_redditadvice_2019.py"
    )
    flags.DEFINE_integer(
        name="max_time_diff",
        default=None,
        help="Maximum time difference between answer pairs in seconds"
    )
    flags.DEFINE_float(
        name="max_len_ratio",
        default=None,
        help="Maximum length ratio between longest and shortest answer in a pair"
    )
    flags.DEFINE_float(
        name="min_score_ratio",
        default=None,
        help="Minimum score ratio between highest and lowest scoring answers in pair"
    )
    flags.DEFINE_integer(
        name="n_datasets",
        default=1,
        help="Build n independent datasets"
    )
    return flags.FLAGS

def to_tsv_line(question, ans1, ans2):
    """
    Creates a tsv line from an answer pair.
    - The latter answer in the line is the one with the higher score.
    - Only question selftext is trimmed to <=1250 tokens, as per Rowan's 
      preprocessing of RedditAdvice2019.

    Args:
    question : dict
        A line from the dataset resulting from data.create_redditadvice_2019.
    ans1 : dict
        An element of question["good_comments"].
    ans2 : dict
        An element of question["good_comments"].
    
    Returns:
    line : str
        A tab-separated line with fields [subreddit, date, title, selftext,
        ans1.body, ans2.body].
    """
    dt_date = datetime.utcfromtimestamp(question["created_utc"])
    str_date = [
        'January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'
        ][dt_date.month - 1] \
        + ' {}, {}'.format(dt_date.day, dt_date.year)
    # Sort the answers by score
    if ans1["score"] > ans2["score"]:
        ans1, ans2 = ans2, ans1
    inputs = "Subreddit: {} Date: {} Title: {} Selftext: {}".format(
        _fix_reddit_text(question["subreddit"]),
        _fix_reddit_text(str_date),
        _fix_reddit_text(question["title"]),
        _fix_reddit_text(_trim_to_desired_length(
            encoder,
            question["selftext"],
            desired_len=SELFTEXT_DESIRED_LEN
        ))
    )
    return "\t".join([
        inputs,
        _fix_reddit_text(ans1["body"]),
        _fix_reddit_text(ans2["body"])
    ])

if __name__ == "__main__":
    FLAGS = _define_flags()
    FLAGS(sys.argv)
    # Assign dataset ids
    if FLAGS.n_datasets > 1:
        id_base = FLAGS.dataset_id or int(time())
        dataset_ids = [f"{id_base}-{i + 1}" for i in range(FLAGS.n_datasets)]
    else:
        dataset_ids = [FLAGS.dataset_id or int(time())]
    # Create dataset directories and store parameters
    for dataset_id in dataset_ids:
        out_dir = os.path.dirname(META_OUT_PATH.format(dataset_id=dataset_id))
        os.makedirs(out_dir, exist_ok=False)
    n_questions = {dataset_id: 0 for dataset_id in dataset_ids}
    n_ans_pairs = {dataset_id: 0 for dataset_id in dataset_ids}
    with open(FLAGS.jsonl_path, "r") as jsonl_file, ExitStack() as stack:
        # Open all dataset files at the same time
        dataset_files = {
            dataset_id: {
                split: stack.enter_context(open(
                    LOCAL_TSV_PATH.format(dataset_id=dataset_id, split=split),
                    "w"
                ))
                for split in SPLITS
            }
            for dataset_id in dataset_ids
        } # dataset_files["id"]["train"] := train split file of dataset "id"
        # Randomly place the questions into the n datasets
        for line in tqdm(jsonl_file):
            question = json.loads(line)
            question_counted = False
            if question_is_valid(question):
                # Which dataset will we store this question in?
                dataset_id = choice(dataset_ids)
                out_file = dataset_files[dataset_id][question["split"]]
            else:
                continue
            for ans1_idx, ans1 in enumerate(question["good_comments"]):
                if not answer_is_valid(ans1):
                    continue
                for ans2 in question["good_comments"][ans1_idx + 1:]:
                    if not answer_is_valid(ans2):
                        continue
                    # Answers and question valid, is answer pair valid?
                    if answer_pair_is_valid(
                        ans1, ans2,
                        max_time_diff=FLAGS.max_time_diff,
                        max_len_ratio=FLAGS.max_len_ratio,
                        min_score_ratio=FLAGS.min_score_ratio
                        ):
                        dataset_line = to_tsv_line(question, ans1, ans2)
                        out_file.write(dataset_line + "\n")
                        n_ans_pairs[dataset_id] += 1
                        if not question_counted:
                            n_questions[dataset_id] += 1
                            question_counted = True
    # Write dataset metadata
    for dataset_id in dataset_ids:
        with open(META_OUT_PATH.format(dataset_id=dataset_id), "w") as meta_f:
            metadata = {
                "dataset_id": dataset_id,
                "n_datasets": FLAGS.n_datasets,
                "jsonl_path": FLAGS.jsonl_path,
                "max_time_diff": FLAGS.max_time_diff,
                "max_len_ratio": FLAGS.max_len_ratio,
                "min_score_ratio": FLAGS.min_score_ratio,
                "n_questions": n_questions[dataset_id],
                "n_ans_pairs": n_ans_pairs[dataset_id]
            }
            json.dump(metadata, meta_f, indent=2)