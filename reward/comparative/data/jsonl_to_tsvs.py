import os
from posixpath import split
import sys
import json
from tqdm import tqdm
from datetime import datetime

from data.assertions import question_is_valid, answer_is_valid
from data.to_tfrecord_t5 import encoder, _trim_to_desired_length, _fix_reddit_text

TSV_PATH = os.path.dirname(__file__) + "/{split}_str.tsv"
SELFTEXT_DESIRED_LEN = 1250
TSV_COLNAMES = ["inputs", "targets1", "targets2"]

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
    if not question_is_valid(question) \
        or not answer_is_valid(ans1) \
        or not answer_is_valid(ans2) \
        or ans1["score"] == ans2["score"]:
        raise ValueError("Answer pair invalid")
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
    """
    Args:
    jsonl_path : str
        Dataset generated by create_redditadvice_2019.py
    """
    jsonl_path = sys.argv[1]
    answer_pairs = {} # {ans_id: [ans_id]}
    with open(jsonl_path, "r") as jsonl_file,\
        open(TSV_PATH.format(split="train"), "w") as train_anss_f,\
        open(TSV_PATH.format(split="val"), "w") as val_anss_f,\
        open(TSV_PATH.format(split="test"), "w") as test_anss_f:
        split_to_file = {
            "train": train_anss_f,
            "val": val_anss_f,
            "test": test_anss_f
        }
        for line in tqdm(jsonl_file):
            question = json.loads(line)
            if not question_is_valid(question):
                continue
            for ans1_idx, ans1 in enumerate(question["good_comments"]):
                if not answer_is_valid(ans1):
                    continue
                for ans2 in question["good_comments"][ans1_idx + 1:]:
                    if not answer_is_valid(ans2):
                        continue
                    # Answers and question valid
                    try:
                        dataset_line = to_tsv_line(question, ans1, ans2)
                        split_file = split_to_file[question["split"]]
                        split_file.write(dataset_line + "\n")
                    except:
                        pass