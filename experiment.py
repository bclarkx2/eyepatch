#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import json
import argparse
import numpy as np
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# LOCAL
from file import top_files
from database import DB
from classifier import classifier_by_name, classifier_names
from classifier import MajorityClassifier, Perceptron, NearestNeighbor
from nlp import fv_by_name, fv_names as feature_vector_names


###############################################################################
# Constants                                                                   #
###############################################################################

NUM_FILES = 10


###############################################################################
# Experiments                                                                 #
###############################################################################

def do_test(classifier_class, fv_class, db, filename, extra_args):

    print("For file {filename}, classifier {classifier}({params}), and fv {fv}".format(
        filename=filename, classifier=classifier_class.__name__, fv=fv_class.__name__, params=extra_args)
    )
    stats = classifier_class.analyze(db, filename, fv_class(db), extra_args, use_cv=True)

    print(stats)
    return stats


def compare_files(db, fv_class, filenames):

    for filename_idx, filename in enumerate(filenames):

        majority_stats = MajorityClassifier.analyze(db,
                                                    filename,
                                                    fv_class(db),
                                                    tuned_args["MajorityClassifier"],
                                                    use_cv=True)

        perceptron_stats = Perceptron.analyze(db,
                                              filename,
                                              fv_class(db),
                                              tuned_args["Perceptron"],
                                              use_cv=True)

        nn_stats = NearestNeighbor.analyze(db,
                                           filename,
                                           fv_class(db),
                                           tuned_args["NearestNeighbor"],
                                           use_cv=True)

        print(filename)
        print("Maj")
        print("Accuracy: {:0.3f}".format(majority_stats.accuracy().avg))
        print("Precision: {:0.3f}".format(majority_stats.precision().avg))
        print("Recall: {:0.3f}".format(majority_stats.recall().avg))
        print()

        print("Per")
        print("Accuracy: {:0.3f}".format(perceptron_stats.accuracy().avg))
        print("Precision: {:0.3f}".format(perceptron_stats.precision().avg))
        print("Recall: {:0.3f}".format(perceptron_stats.recall().avg))
        print()

        print("nn")
        print("Accuracy: {:0.3f}".format(nn_stats.accuracy().avg))
        print("Precision: {:0.3f}".format(nn_stats.precision().avg))
        print("Recall: {:0.3f}".format(nn_stats.recall().avg))
        print()
        print()


###############################################################################
# Helper functions                                                            #
###############################################################################

def pretty(msg):
    return json.dumps(msg, indent=3, sort_keys=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-vector", "--fv", "-f", "-v",
                        choices=feature_vector_names(),
                        required=True,
                        help="Choice of FV representations")
    parser.add_argument("--database",
                        default="eyepatch.db",
                        help="Database file to read from")
    parser.add_argument("--files",
                        default=None,
                        action="append",
                        help="Explicit filenames to test on")
    parser.add_argument("--classifier", "-c",
                        choices=classifier_names(),
                        required=True,
                        help="Choice of classifiers")
    parser.add_argument("--compare",
                        action="store_true")
    parser.add_argument("--tune", "-t",
                        action="store_true")
    return parser.parse_known_args()


def plot_tune_nearest_neighbors(args, extra_args):

    for file in args.files:

        all_stats = []
        all_args = []

        tunables = tunable_map["NearestNeighbor"]

        k_s = np.array(tunables["--k"])
        pos_weights = np.array(tunables["--pos-weight"])

        x_axis = np.transpose(np.array([pos_weights, ] * len(k_s), dtype=float))
        y_axis = np.array([k_s, ] * len(pos_weights), dtype=float)
        recalls = np.zeros(x_axis.shape, dtype=float)

        for extra_arg_str in extra_args:
            stats = do_test(
                    classifier_by_name(args.classifier),
                    fv_by_name(args.feature_vector),
                    db, file, extra_arg_str
            )
            all_stats.append(stats)
            all_args.append(extra_arg_str)

            print("args: {}".format(extra_arg_str))

            indices = get_indicies_for_tuning("NearestNeighbor", extra_arg_str)

            print("indices: {}".format(indices))

            _, recall, _ = stats.recall()

            print("recall: {}".format(recall))

            recalls[indices[0], indices[1]] = recall

            print("recalls: ")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x_axis, y_axis, recalls)
        ax.set_title("Nearest Neighbor Tuning Parameters")
        ax.set_xlabel("pos_weights")
        ax.set_ylabel("k")
        ax.set_zlabel("Recall")

        fig.set_size_inches(9, 6)
        plt.savefig('nearest_neighbors_tuning.png', bbox_inches="tight")


###############################################################################
# Main script                                                                 #
###############################################################################
tunable_map = {
    "NearestNeighbor": {
        "--k": [str(i) for i in range(3, 11)],
        "--pos-weight": [str(0.1 * i) for i in range(1, 6)]
    },
    "Perceptron": {
        "--eta0": [str(0.5 * i) for i in range(1,21)],
        "--class-weight": ["uniform", "balanced"],
    },
    "MajorityClassifier":{},
}

tuned_args = {
    "NearestNeighbor": ["--k", "3", "--pos-weight", "0.1"],
    "Perceptron": ["--eta0", "2.0", "--class-weight", "balanced"],
    "MajorityClassifier": {},
}

def get_indicies_for_tuning(classifier_name, extra_arg_list):
    extra_arg_map = dict(zip(extra_arg_list[0::2], extra_arg_list[1::2]))
    return tuple(tunable_map[classifier_name][key].index(val) for key, val in extra_arg_map.items())


def create_extra_args_for_tuning(classifier_name):
    args = tunable_map[classifier_name].keys()
    values = tunable_map[classifier_name].values()
    args_format_string = " ".join([(arg + " {}") for arg in args])
    return [args_format_string.format(*values_tup).split() for values_tup in product(*values)]


def create_tuned_extra_args(classifier_name):
    args = tuned_args[classifier_name].keys()
    values = tuned_args[classifier_name].values()
    args_format_string = " ".join([(arg + " {}") for arg in args])
    return args_format_string.format(*values)


def main():

    # sys.stderr = StringIO()

    args, extra_args = get_args()

    db = DB(args.database)

    if args.files is None:
        args.files = top_files(db, NUM_FILES)

    if args.tune:
        extra_args = create_extra_args_for_tuning(args.classifier)
        plot_tune_nearest_neighbors(args, extra_args)
    else:
        extra_args = [extra_args]

    if args.compare:
        compare_files(db,
                      fv_by_name(args.feature_vector),
                      args.files)


if __name__ == '__main__':
    main()
