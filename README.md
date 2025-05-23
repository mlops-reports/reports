# experiment

## Description
This is a template for the ml experimentation use cases

## Building the environment
1. Install [zsh](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH) & [oh-my-zsh](https://ohmyz.sh/)
2. Install [poetry](https://python-poetry.org/docs/)
3. Install [heroku](https://devcenter.heroku.com/articles/heroku-cli#install-the-heroku-cli)
4. Run the command(s) below
```
make install_dependencies_linux
make activate_environment
```

You can run "make install_dependencies_win" or make install_dependencies_mac for other OS.

Note that poetry fails to install torch dependencies related to cudNN for torch > 2.0.0
so you might consider to install it via pip.

Lastly, you need to have a CUDA installed on your machine in order to use torch with the
GPU processors. We trained and tested our model with CUDA 12, which can be installed on
Linux machines via:

sudo apt install nvidia-cuda-toolkit

If not installed, the model will solely run on CPU, which might be slow.

## Usage on local machine
create `.env` file in working directory containing required access tokens
```
# general
FERNET_KEY=<FERNET_KEY>
DBT_PROJECT_PATH=<DBT_PROJECT_PATH>

# mlflow
MLFLOW_DB_USERNAME=<DB_HOST>
MLFLOW_DB_PASSWORD=<DB_NAME>
MLFLOW_DB_HOST=<DB_USERNAME>
MLFLOW_DB_NAME="<DB_PASSWORD>"
MLFLOW_S3_ENDPOINT_URL=<MLFLOW_S3_ENDPOINT_URL>
MLFLOW_AWS_BUCKET=<AWS_BUCKET>

MLFLOW_TRACKING_USERNAME=<MLFLOW_TRACKING_USERNAME>
MLFLOW_PROJECT_ROOT=<PATH_TO_THIS_PROJECT>

AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>


# label studio
DJANGO_DB=<DJANGO_DB>
POSTGRE_NAME=<POSTGRE_NAME>
POSTGRE_HOST=<POSTGRE_HOST>
POSTGRE_USER=<POSTGRE_USER>
POSTGRE_PASSWORD=<POSTGRE_PASSWORD>
POSTGRE_PORT=<POSTGRE_PORT>
DISABLE_SIGNUP_WITHOUT_LINK=<DISABLE_SIGNUP_WITHOUT_LINK>
LABEL_STUDIO_USERNAME=<LABEL_STUDIO_USERNAME>
LABEL_STUDIO_PASSWORD=<LABEL_STUDIO_PASSWORD>
LABEL_STUDIO_API_KEY=<LABEL_STUDIO_API_KEY>


# db connection
PSQL_DB_USERNAME=<PSQL_DB_USERNAME>
PSQL_DB_PASSWORD=<PSQL_DB_PASSWORD>
PSQL_DB_HOST=<PSQL_DB_HOST>


# ai
HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN>
OPEN_AI_API_KEY=<OPEN_AI_API_KEY>


# optional
LOG_FILE_PATH=<LOG_FILE_PATH>
LOG_LEVEL=<LOG_LEVEL>
```
## External storage of metadata and artifacts

* [Backend Store (PostgreSQL)](https://mlflow.org/docs/latest/tracking.html#id77)
* [Artifact Store (S3)](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)

## Deployment with docker-compose
https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

## Running Label Studio
label-studio -db postgresql

## Dummy Data Generator
https://mockaroo.com/

## Classes
["Emergency"] = 0
["Normal"] = 1
["Non Emergency [No Doctor]"] = 2
["Non Emergency [Doctor]"] = 3

