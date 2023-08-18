POETRY := poetry

install:
	$(POETRY) env use python3.11;
	$(POETRY) install 