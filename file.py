#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

import argparse
import random
import numpy as np
from operator import itemgetter
from database import DB


###############################################################################
# Constants                                                                   #
###############################################################################

NUM_FOLDS = 5
EXCLUDE_KEYWORDS = [
    "CHANGELOG"
]


###############################################################################
# File selectors                                                              #
###############################################################################

def top_files(db, number):

    files_by_num_issues = filenames_sorted_by_issue(db)

    # get the <number> most interesting files
    interesting_files = files_by_num_issues[-number:]

    return interesting_files


def stratified_files(db, number):

    raise NotImplementedError("this doesn't work yet")

    # files_by_num_issues = files_sorted_by_issue(db)

    # max_involved = max(x[1] for x in files_by_num_issues)

    # cutoffs = range(0, int(max_involved), int(max_involved / NUM_FOLDS))

    # chunks = [] * NUM_FOLDS

    # # import pdb; pdb.set_trace()

    # for filename, num_issues in reversed(files_by_num_issues):

    #     cutoff_idx = int(num_issues / (max_involved / NUM_FOLDS))
    #     print(cutoff_idx)

    #     # for cutoff_idx, cutoff in enumerate(reversed(cutoffs)):
    #     #     if num_issues > cutoff:
    #     #         chunks[cutoff_idx - 1].append(filename)
    #     #         break

    # return chunks

    # chunks = np.array_split(files_by_num_issues, NUM_FOLDS)

    # num_per_chunk = int(number / NUM_FOLDS)

    # stratified = []

    # for chunk in chunks:
    #     for _ in range(num_per_chunk):
    #         stratified.append(random.choice(chunk))

    # return stratified


###############################################################################
# Helper functions                                                            #
###############################################################################

def num_issues(db, file):
    labels = db.labels(file)
    return sum(labels)


def files_sorted_by_issue(db):
    files = db.files()

    issue_count = {file: num_issues(db, file)
                   for file in files
                   if not should_exclude(file)}

    # sort by number of issues involved
    issue_count = sorted(issue_count.items(), key=itemgetter(1))

    return issue_count


def should_exclude(key):
    return any(exclude_word in key for exclude_word in EXCLUDE_KEYWORDS)


def filenames_sorted_by_issue(db):
    files_by_issue = files_sorted_by_issue(db)
    filenames_by_issue = [x[0] for x in files_by_issue]
    return filenames_by_issue


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("number",
                        type=int,
                        help="Number of files to select")
    parser.add_argument("algorithm",
                        choices=["top", "stratified"],
                        help="method to select interesting files")
    parser.add_argument("--database",
                        default="eyepatch.db")
    return parser.parse_args()


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    args = get_args()

    db = DB(args.database)

    if args.algorithm == "top":
        files = top_files(db, args.number)
    elif args.algorithm == "stratified":
        files = stratified_files(db, args.number)

    print(files)


if __name__ == '__main__':
    main()
