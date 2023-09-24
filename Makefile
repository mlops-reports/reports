# Defines variables
POETRY := poetry

LABEL_STUDIO_HOST := "https://label.drgoktugasci.com"

HEROKU := heroku
HEROKU_LABEL_APP_NAME := "label-reports"


# Installs the dependencies
install_dependencies:
	$(POETRY) env use $(shell which python3.11);
	$(POETRY) install

# Activates poetry environment
activate_environment:
	$(POETRY) shell

# Removes the existing environment
remove_environment:
	rm -rf .venv;

setup_label_server_dependencies:
	$(HEROKU) run pip install -r requirements-label.txt --app $(HEROKU_LABEL_APP_NAME);

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


