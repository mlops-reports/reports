# Define variables
POETRY := poetry
POETRY_VERSION := 1.5.1
PYTHON_PATH := $(shell which python3.11)
LABEL_STUDIO_HOST := "https://label.drgoktugasci.com"
HEROKU_APP_NAME := "er-reports"

# Install the poetry environment
install:
	rm -rf .venv;
	curl -sSL https://install.python-poetry.org | python3 - --version $(POETRY_VERSION); 
	$(POETRY) env use $(PYTHON_PATH);
	$(POETRY) install
	
# Starts the Label Studio server
start_labeling_server:
	heroku ps:scale web=1 --app $(HEROKU_APP_NAME);
	open $(LABEL_STUDIO_HOST)/projects/

# Stops the Label Studio server
stop_labeling_server:
	heroku ps:scale web=0 --app $(HEROKU_APP_NAME)

