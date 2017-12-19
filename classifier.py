#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import sklearn.linear_model
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import cross_validate
from collections import namedtuple
from operator import attrgetter
import argparse

# LOCAL
from database import DB
from stats import ManualStats, ScikitStats

from nlp import fv_by_name

###############################################################################
# Constants                                                                   #
###############################################################################

METRICS = ['accuracy', 'precision', 'recall', 'roc_auc']
NUM_FOLDS = 5


###############################################################################
# Classes                                                                     #
###############################################################################

Data = namedtuple('Data', ('issue_ids', 'fvs', 'labels'))


class Classifier(object):
    """Base class for all eyepatch classifiers

    Provides basic interface for:
        1. Directions to get training data and labels from db
        2. Using the trained classifier
    """

    def __init__(self, db):
        super(Classifier, self).__init__()
        self.db = db

    @staticmethod
    def analyze(db, filename, fv_obj, extra_args, use_cv=False):
        pass

    # Decide whether this issue_id involves this classifier's file
    def classify(self, issue_id):
        raise NotImplementedError("Implement in subclass, please")


    @staticmethod
    def parse_extra_args(extra_args):
        raise NotImplementedError("Implement in subclass")

    # A nice evaluate method that computs the confusion matrix given:
    #   1. A list of issue ids to test
    #   2. A list of labels to check them against
    #   3. The existance of the classify method
    # Note that this method would be performed _after_ cv
    def evaluate(self, issue_ids, labels):
        TP = FP = TN = FN = 0
        for issue_id, label in zip(issue_ids, labels):
            found_class = self.classify(issue_id)

            if label == 1.0:
                if found_class == 1.0:
                    TP += 1
                else:
                    FN += 1
            else:
                if found_class == 1.0:
                    FP += 1
                else:
                    TN += 1

        return TP, FP, TN, FN

    # Does the same as the above method, but groups everything into
    # a nice Stats object
    def evaluate_stats(self, issue_ids, labels):
        TP, FP, TN, FN = self.evaluate(issue_ids, labels)
        return ManualStats([TP], [FP], [TN], [FN])


class MajorityClassifier(Classifier):
    """Classifier that simply returns the majority class"""

    def __init__(self, db, labels):
        super(MajorityClassifier, self).__init__(db)

        num = len(labels)
        num_pos = sum(labels)
        num_neg = num - num_pos

        if num_pos >= num_neg:
            self.majority_class = 1.0
        else:
            self.majority_class = 0.0

    @staticmethod
    def analyze(db, filename, fv_obj, extra_args, use_cv=False):

        issue_ids = db.issue_ids()
        fvs = fvs = db.feature_vectors(fv_obj)
        labels = db.labels(filename)

        MajorityClassifier.parse_extra_args(extra_args)

        if use_cv:
            stats = ManualStats([], [], [], [])
            for train, test in partition(issue_ids, fvs, labels):
                mc = MajorityClassifier(db, train.labels)
                stats.update(*mc.evaluate(test.issue_ids, test.labels))
            return stats
        else:
            nn = MajorityClassifier(db, labels)
            return nn.evaluate_stats(issue_ids, labels)

    @staticmethod
    def parse_extra_args(extra_args):
        parser = argparse.ArgumentParser()
        return parser.parse_args(extra_args)

    def classify(self, issue_id):
        return self.majority_class


class Perceptron(Classifier):
    """Perceptron classifier"""

    def __init__(self, db, fv_obj, class_weight, eta0):
        super(Perceptron, self).__init__(db)
        self.fv_obj = fv_obj
        if class_weight == "uniform":
            class_weight = None
        self.perceptron = sklearn.linear_model.Perceptron(penalty='l2',
                                                          class_weight=class_weight,
                                                          eta0=eta0,
                                                          max_iter=1000,
                                                          tol=1e-3)

    def fit(self, *args, **kwargs):
        return self.perceptron.fit(*args, **kwargs)

    def cross_validate(self, feature_vectors, labels):
        scores = cross_validate(self.fit(feature_vectors, labels),
                                feature_vectors,
                                labels,
                                cv=NUM_FOLDS,
                                scoring=METRICS)
        return ScikitStats(scores)

    @staticmethod
    def parse_extra_args(extra_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--class-weight","-w",
                            choices=["uniform", "balanced"],
                            default="uniform"
                            )
        parser.add_argument("--eta0", "-e",
                            type=float,
                            default=1.0)
        return parser.parse_args(extra_args)


    @staticmethod
    def analyze(db, filename, fv_obj, extra_args, use_cv=False):

        feature_vectors = db.feature_vectors(fv_obj)
        labels = db.labels(filename)
        issue_ids = db.issue_ids()
        extra_args_ns = Perceptron.parse_extra_args(extra_args)

        perceptron = Perceptron(db, fv_obj, class_weight=extra_args_ns.class_weight, eta0=extra_args_ns.eta0)

        if use_cv:
            return perceptron.cross_validate(feature_vectors, labels)
        else:
            perceptron.fit(feature_vectors, labels)
            return perceptron.evaluate_stats(issue_ids, labels)

    def classify(self, issue_id):
        fv = self.db.feature_vector(self.fv_obj, issue_id)

        # Must reshape feature because it's only one sample
        shaped_fv = np.reshape(fv, (1, -1))

        prediction_arr = self.perceptron.predict(shaped_fv)

        # get actual class out
        prediction = prediction_arr.item(0)
        return prediction


class NearestNeighbor(Classifier):
    """Classifier implementation that uses Nearest Neighbor algorithm

    Can be set to weight positive classes and use up to k nearest neighbors.
    """

    Issue = namedtuple('Issue', ('label', 'distance'))

    def __init__(self, db, fv_obj, fvs, labels, k=3, pos_weight=0.5):
        super(NearestNeighbor, self).__init__(db)
        self.fv_obj = fv_obj
        self.fvs = fvs
        self.labels = labels
        self.k = k
        self.pos_weight = pos_weight
        self.issues = [NearestNeighbor.Issue(label, fv) for label, fv in zip(labels, fvs)]

    @staticmethod
    def distance(fv_from, fv_to):
        return euclidean(fv_from, fv_to)

    @staticmethod
    def parse_extra_args(extra_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--k", "-k",
                            type=int, default=3)
        parser.add_argument("--pos-weight", "-p",
                            type=float)
        args_ns = parser.parse_args(extra_args)
        if args_ns.pos_weight is None:
            args_ns.pos_weight = 1/args_ns.k
        return args_ns

    @staticmethod
    def analyze(db, filename, fv_obj, extra_args, use_cv=False):

        issue_ids = db.issue_ids()
        fvs = db.feature_vectors(fv_obj)
        labels = db.labels(filename)
        extra_args_ns = NearestNeighbor.parse_extra_args(extra_args)

        if use_cv:
            stats = ManualStats([], [], [], [])
            for train, test in partition(issue_ids, fvs, labels):
                nn = NearestNeighbor(db, fv_obj,
                                     train.fvs, train.labels,
                                     k=extra_args_ns.k,
                                     pos_weight=extra_args_ns.pos_weight)
                stats.update(*nn.evaluate(test.issue_ids, test.labels))
            return stats
        else:
            nn = NearestNeighbor(db, fv_obj, fvs, labels,
                                 k=extra_args_ns.k,
                                 pos_weight=extra_args_ns.pos_weight)
            return nn.evaluate_stats(issue_ids, labels)

    def classify(self, issue_id_from):
        fv_from = self.db.feature_vector(self.fv_obj, issue_id_from)
        return self.classify_fv(fv_from)

    def classify_fv(self, fv_from):

        neighbors = []
        prediction = 0.0

        for fv_to, label_to in zip(self.fvs, self.labels):
            dist = self.distance(fv_from, fv_to)
            neighbor = NearestNeighbor.Issue(label_to, dist)
            neighbors.append(neighbor)

        sorted_neighbors = sorted(neighbors, key=attrgetter('distance'))

        k_nearest = sorted_neighbors[:self.k]

        num_pos = sum(issue.label for issue in k_nearest)
        frac_pos = num_pos / self.k

        if frac_pos >= self.pos_weight:
            prediction = 1.0
        else:
            prediction = 0.0

        # print("fv: {}; {}; {}".format(fv_from, frac_pos, prediction))
        # print("   k_nearest: {}".format(k_nearest))
        return prediction


###############################################################################
# Helper functions                                                            #
###############################################################################

def partition(issue_ids, fvs, labels, num_folds=NUM_FOLDS):

    indices = list(range(len(issue_ids)))
    index_set = set(indices)

    neg_examples = [index for index in indices if labels[index] == 0.0]
    pos_examples = [index for index in indices if labels[index] == 1.0]

    neg_folds = np.array_split(neg_examples, num_folds)
    pos_folds = np.array_split(pos_examples, num_folds)

    folds = [set(np.concatenate((neg_fold, pos_fold)))
             for neg_fold, pos_fold
             in zip(neg_folds, pos_folds)]

    for fold in folds:
        train_indices = index_set - fold
        test_indices = fold

        train_issue_ids = [issue_ids[idx] for idx in train_indices]
        train_fvs = [fvs[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]
        train = Data(train_issue_ids, train_fvs, train_labels)

        test_issue_ids = [issue_ids[idx] for idx in test_indices]
        test_fvs = [fvs[idx] for idx in test_indices]
        test_labels = [labels[idx] for idx in test_indices]
        test = Data(test_issue_ids, test_fvs, test_labels)

        yield train, test

def classifier_names():
    return [x.__name__ for x in Classifier.__subclasses__()]


def classifier_by_name(name):
    for classifier in Classifier.__subclasses__():
        if name == classifier.__name__:
            return classifier


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    db = DB("eyepatch.db")

    issue_ids = list(range(20))
    fvs = [[idx] for idx in range(20)]
    labels = [1.0] * 10 + [0.0] * 10

    for train, test in partition(issue_ids, fvs, labels):
        print("{}".format(train))
        print("{}".format(test))
        print()

    from unittest.mock import MagicMock
    fvs = [[0, 0], [2, 2], [10, 10], [12, 12], [20, 20], [21, 21], [22, 22]]
    labels = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    nn = NearestNeighbor(db, MagicMock(), fvs, labels, k=4, pos_weight=0.2)

    # positive
    nn.classify_fv([11, 11])

    # negative
    nn.classify_fv([1, 1])

    nn.classify_fv([16, 16])

    nn.classify_fv([25, 25])


if __name__ == '__main__':
    main()
