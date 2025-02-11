# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
import os
import time
from tqdm import tqdm

import random
from collections import defaultdict


def extract_judgment(judgment):
    if "[[A]]" in judgment:
        return "A"
    elif "[[B]]" in judgment:
        return "B"
    else:
        return 'A' if random.random() < 0.5 else 'B'

def compute_acc(args):
    accs = defaultdict(list)
    random.seed(123)
    for line in open(args.answers_file):
        ex = json.loads(line)
        pred = extract_judgment(ex['output'])
        acc = int(pred == ex['Label'])
        accs['all'].append(acc)
        category = ex['Meta']['Category']
        if category == 'safety':
            if ex['ID'].lower().startswith('pairs'):
                category = 'safety/bias'
            else:
                category = 'safety/toxicity'
        elif category == 'reasoning':
            if ex['ID'].lower().startswith('math'):
                category = 'reasoning/math'
            else:
                category = 'reasoning/coding'
        accs[category].append(acc)
    for task in accs:
        print(f"acc {task}: {sum(accs[task])} / {len(accs[task])} = {sum(accs[task])/len(accs[task])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="outputs/gpt4o.jsonl")
    args = parser.parse_args()

    compute_acc(args)

"""
cd <repo>

python scripts/2_get_accuracy.py
"""


