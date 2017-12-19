#!/usr/bin/env python3
from get_issues import ReferencedPullHTMLParser
import requests

url= "https://github.com/rails/rails/issues/29219"


resp = requests.get(url)
parser = ReferencedPullHTMLParser()
parser.feed(resp.text)
print(parser.maybe_pr_number, parser.last_pr_ref)
