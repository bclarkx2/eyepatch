#! /usr/bin/env python3

import sqlite3


class DB(object):
    """accessor object for DB

    Maintains a db connection for the life of the object. Useful
    for things?
    """

    def __init__(self, db_file):
        super(DB, self).__init__()
        self.conn = sqlite3.connect(db_file)
        self.c = self.conn.cursor()

    def close(self):
        self.conn.close()

    def execute(self, cmd, args=()):
        return self.c.execute(cmd, args)

    def execute_lst(self, cmd, args=()):
        return _remove_tuples(self.execute(cmd, args))

    def issue_files(self, issue):
        issue_query = """
            SELECT Usage.filename
            FROM issues as Issue, usages as Usage
            WHERE
                Issue.issue_id = ? AND
                Issue.pr_id = Usage.pr_id
        """

        files = self.execute_lst(issue_query, (issue,))

        return files

    def issues(self):
        query = """
            SELECT issue_id, text
            FROM issues
        """
        return self.execute(query)

    def issue_ids(self):
        return [x[0] for x in self.issues()]

    def feature_vector(self, fv_obj, issue_id):
        query = """
            SELECT *
            FROM {}
            WHERE issue_id = ?
        """.format(fv_obj.table_name())

        fv = self.execute(query, (issue_id,)).fetchone()
        without_id = list(fv[1:])
        return without_id

    def feature_vectors(self, fv_obj):
        # apparently can't use SQLite params for column names?
        query = """
            SELECT *
            FROM {}
        """.format(fv_obj.table_name())

        fvs = []
        for fv in self.execute(query):
            without_id = list(fv[1:])
            fvs.append(without_id)

        return fvs

    def files(self):
        query = """
            SELECT filename
            FROM files
        """
        return self.execute_lst(query)

    def extract_features(self, feature_vector):

        fv = feature_vector

        self.execute(fv.create_table_query())

        for issue_id, body in list(self.issues()):

            insert_query = fv.insert_query()
            insert_vals = (issue_id,) + fv.features(body)

            print("insert_vals: {}".format(insert_vals))

            self.execute(insert_query, insert_vals)

        self.commit()

    def labels(self, filename):
        query = """
            SELECT Issue.issue_id
            FROM issues as Issue, usages as Usage
            WHERE
                Usage.filename = ? AND
                Issue.pr_id = Usage.pr_id
        """

        involved_issues = self.execute_lst(query, (filename,))

        labels = [label(issue_id in involved_issues) for issue_id, _ in self.issues()]

        return labels

    def remove_features(self, feature_vector):
        self.execute(feature_vector.drop_query())

    def commit(self):
        self.conn.commit()

    # def commitable(self, f):
    #     def commit_wrapper():
    #         result = f()
    #         self.conn.c


def _remove_tuples(lst):
    return [x[0] for x in lst]


def label(is_positive):
    if is_positive:
        return 1.0
    else:
        return 0.0
