#! /usr/bin/env python3
"""serialize_issues"""
import sqlite3
from json import load
from sys import argv


CREATE_ISSUES_DDL = """
CREATE TABLE IF NOT EXISTS issues (
    text TEXT,
    issue_id INT PRIMARY KEY,
    pr_id INT
)
"""

INSERT_ISSUE_SQL = """
INSERT INTO issues(issue_text, issue_id, pr_id)
VALUES (?, ?, ?)
"""
INSERT_ISSUE_SQL = """
    INSERT INTO issues(text, issue_id, pr_id) VALUES (?, ?, ?)
"""

SETUP_DDL = CREATE_ISSUES_DDL

def extract_fields_gen(issues):
    for issue in issues:
        yield (issue["body"], issue["number"], pr_id(issue))

def pr_id(issue):
    pr_id_str = issue.get("closed_by")
    if not pr_id_str:
        return None
    try:
        return int(pr_id_str)
    except ValueError:
        return None

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
