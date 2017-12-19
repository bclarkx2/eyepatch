#! /usr/bin/env python3
"""serialize_pulls"""
import sqlite3
from json import load
from sys import argv


CREATE_PULLS_DDL = """
    CREATE TABLE IF NOT EXISTS pulls (
        id INT PRIMARY KEY,
    )
"""

INSERT_ISSUE_SQL = """
    INSERT INTO pulls (id) VALUES (?)
"""


SETUP_DDL = CREATE_ISSUES_DDL

def extract_fields_gen(pulls):
    for pull in pulls:
        yield (pull["number"],)

def main():
    conn = sqlite3.connect("../eyepatch.db")
    c = conn.cursor()

    c.execute(SETUP_DDL)

    with open(argv[1], 'rt') as f:
        issues = load(f)

    interesting_fields = extract_fields_gen(issues)

    c.executemany(INSERT_ISSUE_SQL, interesting_fields)
    conn.commit()


if __name__ == '__main__':
    main()
