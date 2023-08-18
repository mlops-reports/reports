POETRY := poetry

install:
	curl -sSL https://install.python-poetry.org | python3 - --version 1.5.1; 
	$(POETRY) env use python3.11;
	$(POETRY) install 