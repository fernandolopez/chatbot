MESOP=poetry run mesop

run:
	$(MESOP) main.py

setup:
	poetry install

.PHONY: run setup
