POETRY := poetry
POETRY_VERSION := 1.5.1
PYTHON_PATH := $(shell which python3.11)

install:
	rm -rf .venv;
	curl -sSL https://install.python-poetry.org | python3 - --version $(POETRY_VERSION); 
	$(POETRY) env use $(PYTHON_PATH);
	$(POETRY) install 