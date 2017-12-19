#!/usr/bin/env python3

import requests
import json
import sys
import argparse
import sqlite3
from sys import argv
from html.parser import HTMLParser

host = "https://api.github.com"
repo = "repos/rails/rails"


class ReferencedPullHTMLParser(HTMLParser):
    """docstring for ReferencedPullHTMLParser"""

    def __init__(self):
        super().__init__()
        self.in_disc_item = -1
        self.closer_is_next = False
        self.maybe_pr_number = None
        self.last_pr_ref = None
        self.disc_item_merged = False
        self.pr_ref_next = False

    def handle_starttag(self, tag, attrs):
        if self.in_disc_item >= 0:
            self.in_disc_item += 1
            # eprint("past closer ({})".format(self.in_disc_item))
        else:
            self.disc_item_merged = False
        if ("class", "discussion-item") in attrs or ("class", "discussion-item discussion-item-closed") in attrs:
            # eprint("entering discussion-item")
            self.in_disc_item = 0
        if ("title", "State: merged") in attrs:
            self.disc_item_merged = True
        if all((
                ("class", "issue-num") in attrs,
                self.in_disc_item >= 0,
                self.disc_item_merged
            )):
            self.pr_ref_next = True

    def handle_endtag(self, tag):
        if self.in_disc_item >= 0:
            self.in_disc_item -= 1
            # eprint("past discussion-item ({})".format(self.in_disc_item))

    def handle_data(self, data):
        if self.pr_ref_next:
            self.last_pr_ref = data[1:]
            self.pr_ref_next = False
            eprint("maybe closed by {}".format(self.last_pr_ref))
        if self.closer_is_next:
            self.maybe_pr_number = data[1:]
            self.closer_is_next = False
            eprint("closed by {}".format(self.maybe_pr_number))
        if self.in_disc_item >= 0 and data.split() == ["closed", "this", "in"]:
            # eprint("found closer data node")
            self.closer_is_next = True



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def scrape(auth, resource):

    url_params = {
        "state": "closed",
        "labels": "attached PR",
        "page": 0
    }

    issues = []

    page = 1
    found_data = True

    while found_data:

        url = "{}/{}/{}".format(host, repo, resource)

        url_params["page"] = page

        response = requests.get(url, params=url_params, auth=auth)

        response_json = parse_resouce(resource, response)

        found_data = response.ok and len(response_json) > 0

        issues.extend(response_json)

        eprint("page: {}; found data: {}".format(page, found_data))

        page += 1

    print(json.dumps(issues, indent=3))

def scrape_files(auth):

    tree_params = {"recursive": 1}
    tree_uri = "{}/{}/git/trees/master".format(host, repo)
    tree_response = requests.get(tree_uri, params=tree_params, auth=auth).json()

    tree_contents = tree_response["tree"]

    print(json.dumps(tree_contents, indent=3))


PULL_ID_QUERY = """
SELECT Issue.pr_id
FROM issues as Issue
WHERE pr_id IS NOT NULL
"""

CREATE_USAGES = """
CREATE TABLE IF NOT EXISTS usages (
    pr_id INT,
    filename TEXT,
    FOREIGN KEY(filename) REFERENCES files(filename)
)
"""

INSERT_USAGE = """
INSERT INTO usages (pr_id, filename)
VALUES (?, ?)
"""


def scrape_usages(auth, pull_filename):

    conn = sqlite3.connect("../eyepatch.db")
    c = conn.cursor()

    c.execute(CREATE_USAGES)

    # get all pull ids associated with issues
    pull_ids = [x[0] for x in list(c.execute(PULL_ID_QUERY))]
    for pull_id in pull_ids:

        # retrieve files associated with with this pull id
        pull_files_uri = "{}/{}/pulls/{}/files".format(host, repo, pull_id)
        pull_files = requests.get(pull_files_uri, auth=auth).json()

        try:
            pull_filenames = [x["filename"] for x in pull_files]

            # insert each filename into the usages table
            for filename in pull_filenames:
                c.execute(INSERT_USAGE, (pull_id, filename))

            eprint("Pull: {}".format(pull_id))
        except Exception:
            eprint("FAILED TO INSERT FROM {}".format(pull_id))

    conn.commit()


def scrape_usages_json(auth, pull_filename):
    with open(pull_filename) as pull_file:
        pulls = json.load(pull_file)

    usages = {}

    for pull in pulls:

        number = pull["number"]

        pull_files_uri = "{}/{}/pulls/{}/files".format(host, repo, number)

        pull_files = requests.get(pull_files_uri, auth=auth).json()

        try:
            pull_filenames = [x["filename"] for x in pull_files]
        except Exception:
            pass

        eprint("Pull: {}".format(number))

        usages.update({number: pull_filenames})

    print(json.dumps(usages, indent=3))


def parse_resouce(resource, response):
    if resource == "issues":
        response_json = issues = response.json()
        for issue in issues:
            html_response = requests.get(issue["html_url"])
            parser = ReferencedPullHTMLParser()
            parser.feed(html_response.text)
            issue["closed_by"] = parser.maybe_pr_number
            if issue["closed_by"] is None:
                issue["closed_by"] = parser.last_pr_ref
        return response_json
    elif resource == "pulls":
        response_json = response.json()
        for pull in response_json:
            issue_url = pull['issue_url']
            issue_id = issue_url.split('/')[-1]
            pull["issue_id"] = issue_id
        return response_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("pswd")
    parser.add_argument("resource", choices=["issues", "pulls", "files", "usages"])
    parser.add_argument("--pulls", default="./pulls")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    auth = (args.username, args.pswd)

    if args.resource in ["issues", "pulls"]:
        scrape(auth, args.resource)
    elif args.resource == "files":
        scrape_files(auth)
    elif args.resource == "usages":
        scrape_usages(auth, args.pulls)


# print("uniques: {}".format(len(set(numbers))))

# curl \
#     -H "Accept: application/json" \
#     $host"/repos/rails/rails/issues?"$url_params
