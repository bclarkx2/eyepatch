

all: test

.PHONY: requirements
requirements: requirements.txt
	pip install -r requirements.txt

.PHONY: test
test: requirements
	pytest test_nbayes.py

.PHONY: clean
clean:
	find . -name "__pycache__" -delete
