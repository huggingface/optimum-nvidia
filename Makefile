fix-quality:
	python3 -m ruff check examples scripts src tests --fix
	python3 -m ruff format examples scripts src tests

quality:
	python3 -m ruff check examples scripts src tests
	python3 -m ruff format examples scripts src tests --check