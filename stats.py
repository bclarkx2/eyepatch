
###############################################################################
# Imports                                                                     #
###############################################################################

from collections import namedtuple
import numpy as np


###############################################################################
# Precision                                                                   #
###############################################################################

PRECISION = 3


###############################################################################
# Helper functions                                                            #
###############################################################################

Data = namedtuple('Data', ['series', 'avg', 'std'])


def zero_divisible(division_func):
    def wrapper(*args, **kwargs):
        try:
            return division_func(*args, **kwargs)
        except ZeroDivisionError:
            return float('nan')
    return wrapper


def stats(lst):
    return Data(lst, np.average(lst), np.std(lst))


###############################################################################
# Stats Classes                                                               #
###############################################################################

class Stats(object):
    """Base class for all stats objects

    Provides interface to be used when comparing the results of
    different classifiers.

    It is expected that any metrics described here will be defined
    by any subclass of Classifier
    """

    def __init__(self):
        super(Stats, self).__init__()

    def accuracy():
        raise NotImplementedError("Implement in subclass")

    def precision():
        raise NotImplementedError("Implement in subclass")

    def recall():
        raise NotImplementedError("Implement in subclass")

    def AUROC():
        raise NotImplementedError("Implement in subclass")

    def __str__(self):
        report = "\n".join([
            self._report_string("Accuracy", self.accuracy()),
            self._report_string("Precision", self.precision()),
            self._report_string("Recall", self.recall())
        ])
        return report

    def _report_string(self, name, data):
        return "{name}: {1:0.3f} {2:0.3f}".format(*data, name=name)


class ManualStats(Stats):
    """Stats subclass that builds required statistics manually

    Turns arrays of TP, FP, TN, FN into stats
    """

    def __init__(self, TP, FP, TN, FN):
        super(ManualStats, self).__init__()
        self._TP = TP
        self._FP = FP
        self._TN = TN
        self._FN = FN

    def accuracy(self):
        accuracy = []
        for TP, TN, FP, FN in zip(self._TP, self._TN, self._FP, self._FN):
            N = TP + TN + FP + FN
            accuracy.append(self._accuracy(TP, TN, N))
        return stats(accuracy)

    @zero_divisible
    def _accuracy(self, TP, TN, N):
        return (TP + TN) / N

    def precision(self):
        precision = []
        for TP, FP in zip(self._TP, self._FP):
            precision.append(self._precision(TP, FP))
        return stats(precision)

    @zero_divisible
    def _precision(self, TP, FP):
        return TP / (TP + FP)

    def recall(self):
        recall = []
        for TP, FN in zip(self._TP, self._FN):
            recall.append(self._recall(TP, FN))
        return stats(recall)

    @zero_divisible
    def _recall(self, TP, FN):
        return TP / (TP + FN)

    def total_negative(self):
        total_neg = []
        for TN, FN in zip(self._TN, self._FN):
            total_neg.append(TN + FN)
        return stats(total_neg)

    def total_positive(self):
        total_pos = []
        for TP, FP in zip(self._TP, self._FP):
            total_pos.append(TP + FP)
        return stats(total_pos)

    def update(self, TP, FP, TN, FN):
        self._TP.append(TP)
        self._FP.append(FP)
        self._TN.append(TN)
        self._FN.append(FN)


class ScikitStats(Stats):
    """Stats implementation for scikit learners"""

    def __init__(self, scores):
        super(ScikitStats, self).__init__()
        self.scores = scores

    def accuracy(self):
        return self._eval("test_accuracy")

    def precision(self):
        return self._eval("test_precision")

    def recall(self):
        return self._eval("test_recall")

    def AUROC(self):
        return self._eval("test_roc_auc")

    def _eval(self, stat_name):
        arr = self.scores[stat_name]
        return Data(arr, arr.mean(), arr.std())


class EmptyStats(Stats):
    """Dummy implementation of Stats for testing purposes"""

    def __init__(self):
        super(EmptyStats, self).__init__()

    def accuracy(self):
        return self._empty()

    def precision(self):
        return self._empty()

    def recall(self):
        return self._empty()

    def AUROC(self):
        return self._empty()

    def _empty(self):
        return [], 0.0, 0.0
