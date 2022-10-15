FILE=. 

.PHONY: tests
tests:
	isort ${FILE}
	black ${FILE}
	mypy ${FILE}
	flake8 ${FILE}
	pytest tests/
