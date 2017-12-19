#! /usr/bin/env python3
import argparse
import hashlib
import nltk

from database import DB

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures



class FeatureVector(object):
    """Base class to represent the feature vector of a bug report


    This class provides the interface to define a feature vector from the
    text of a bug report.
    """
    def __init__(self,db):
        super().__init__()
        self._db = db

    def features(self):
        """Returns a tuple of all generated feature vector values"""
        raise NotImplementedError("Subclass must define feature values")

    def columns(self):
        """Returns a list of (name, type) tuples of all the generated features"""
        raise NotImplementedError("Subclass must define column names")

    def table_name(self):
        """Returns the name of the table that should be generated using
        this feature vector representation"""
        raise NotImplementedError("Subclass must define table names")

    def create_table_query(self):
        query = """
            CREATE TABLE IF NOT EXISTS {table_name} (
                issue_id INT PRIMARY KEY,
                {columns},
                FOREIGN KEY(issue_id) REFERENCES issues(issue_id)
            )
        """

        column_entries = ["{0} {1}".format(*col) for col in self.columns()]
        columns_string = ",".join(column_entries)

        query = query.format(table_name=self.table_name(),
                             columns=columns_string)

        return query

    def insert_query(self):
        query = """
            INSERT INTO {table_name} VALUES ({values})
        """

        num_features = len(self.columns())
        num_columns = 1 + num_features  # account for id
        values = ", ".join("?" * num_columns)

        query = query.format(table_name=self.table_name(),
                             values=values)
        return query

    def drop_query(self):
        return "DROP TABLE IF EXISTS {}".format(self.table_name())


class SimpleFV(FeatureVector):
    """A sample feature vector just to prove that the approach works"""

    def features(self, body):
        text = nltk.Text(word_tokenize(body))
        rails_count = text.count("rails")
        num_words = len(text.vocab())
        return (rails_count, num_words)

    def columns(self):
        return [
            ("rails_count", "INT"),
            ("num_words", "INT")
        ]

    @classmethod
    def table_name(cls):
        return "simple_fv"

class BigramFV(FeatureVector):

    def __init__(self, db):
        super().__init__(db)
        self._token_corpus = []
        self._bigram_measures = BigramAssocMeasures()
        for issue in db.issues():
            self._token_corpus += word_tokenize(issue[1])
        corpus_bytes = "".join(self._token_corpus).encode("utf8")
        self._corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
        self._corpus_finder = BigramCollocationFinder.from_words(self._token_corpus)
        self._corpus_finder.apply_word_filter(self.simple_filter)
        self._bigram_scores = dict(self._corpus_finder.score_ngrams(self._bigram_measures.raw_freq)[:20])
        # print("bi scores: ", self._bigram_scores)
        self._col_names = ["_".join(bigram) for bigram in self._bigram_scores.keys()]

    def columns(self):
        return [(col_name, "REAL") for col_name in self._col_names]

    def features(self, body):
        body_finder = BigramCollocationFinder.from_words(word_tokenize(body))
        body_finder.apply_word_filter(self.simple_filter)
        scores = {
            bigram: body_finder.score_ngram(self._bigram_measures.raw_freq, bigram[0], bigram[1])
            for bigram in self._bigram_scores.keys()
        }
        standardized_scores = [
            self.safe_div(scores[bigram], corp_score)
            for bigram, corp_score in self._bigram_scores.items()
        ]
        return tuple(standardized_scores)

    @staticmethod
    def safe_div(num, denom):
        if num is None or denom is None:
            return 0
        return num / denom

    @staticmethod
    def simple_filter(w, stops=set(stopwords.words("english"))):
        return (not w.isalnum()) or (w in stops)

    def table_name(self):
        return "bigram_fv_{}".format(self._corpus_hash)


###############################################################################
# Helper functions                                                            #
###############################################################################

def fv_names():
    return [x.__name__ for x in FeatureVector.__subclasses__()]


def fv_by_name(name):
    for fv in FeatureVector.__subclasses__():
        if name == fv.__name__:
            return fv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_vector",
                        choices=fv_names(),
                        help="The feature vector type to use")
    parser.add_argument("--database",
                        default="eyepatch.db",
                        help="Database file to read from")
    parser.add_argument("--delete", "-d",
                        action="store_true",
                        help="Flag to delete a feature vector table")
    return parser.parse_args()

###############################################################################
# Main script                                                                 #
###############################################################################

def main():
    args = get_args()

    db = DB(args.database)
    feature_vector = fv_by_name(args.feature_vector)(db)

    if args.delete:
        db.remove_features(feature_vector)
    else:
        db.extract_features(feature_vector)


if __name__ == '__main__':
    main()
