POETRY := poetry

install:
	curl -sSL https://install.python-poetry.org | python3 -; 
	$(POETRY) env use python3.11;
	$(POETRY) install 