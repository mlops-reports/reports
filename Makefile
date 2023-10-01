# Defines variables
PYTHON_PATH := $(shell which python3.11)
PIP_REQUIREMENTS := requirements-dev.txt

POETRY := poetry
MAKE := make
PYTHON := python3

LABEL_STUDIO_HOST := "https://label.drgoktugasci.com"

HEROKU := heroku
HEROKU_LABEL_APP_NAME := "label-reports"


# Installs the dependencies
install_dependencies:
	$(MAKE) remove_environment;
	$(HEROKU) update;
	rm -rf .venv;
	$(POETRY) env use $(PYTHON_PATH);
	$(POETRY) install

# Export dependencies for pip install
export_dependencies:
	$(POETRY) export --without-hashes --format=requirements.txt > $(PIP_REQUIREMENTS) --with mlops,ml

# Removes the existing environment
remove_environment:
	rm -rf .venv

# Activates poetry environment
activate_environment:
	$(POETRY) shell

# Installs Heroku and authenticates the user
setup_labeling_server_connection:
	brew tap $(HEROKU)/brew && brew install $(HEROKU);
	$(HEROKU) login;
	$(HEROKU) keys:add
	
# Starts the Label Studio server
start_labeling_server:
	$(HEROKU) ps:scale web=1 --app $(HEROKU_LABEL_APP_NAME);
	open $(LABEL_STUDIO_HOST)/projects/

# Stops the Label Studio server
stop_labeling_server:
	$(HEROKU) ps:scale web=0 --app $(HEROKU_LABEL_APP_NAME)

# Upload tasks
upload_tasks:
	# $(MAKE) activate_environment;
	$(PYTHON) scripts/prepare_ls_tasks.py
