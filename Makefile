POETRY := poetry
PYTHON_PATH := $(shell which python3.11)

install:
	curl -sSL https://install.python-poetry.org | python3 - --version 1.5.1; 
	$(POETRY) env use $(PYTHON_PATH);
	$(POETRY) install 