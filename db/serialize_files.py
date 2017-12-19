#! /usr/bin/env python3
"""serialize_files"""
import sqlite3
from json import load
from sys import argv

CREATE_FILES_DDL = """
    CREATE TABLE IF NOT EXISTS files (
        filename TEXT PRIMARY KEY
    )
"""

INSERT_FILE_SQL = """
    INSERT INTO files(filename) VALUES (?)
"""


SETUP_DDL = CREATE_FILES_DDL


def extract_fields_gen(files):
    for file in files:
        yield (file["path"],)


def main():
    conn = sqlite3.connect("../eyepatch.db")
    c = conn.cursor()

    c.execute(SETUP_DDL)

    with open(argv[1], 'rt') as f:
        files = load(f)

    interesting_fields = extract_fields_gen(files)

    c.executemany(INSERT_FILE_SQL, interesting_fields)
    conn.commit()


if __name__ == '__main__':
    main()
