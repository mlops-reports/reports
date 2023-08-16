# experiment

## Description
This is a template for the ml experimentation use cases

## Creating a Virtual Environment
```
poetry install
```

## Usage on local machine
create `.env` file in working directory containing required access tokens
```
# mlflow
DB_HOST=<DB_HOST>
DB_NAME=<DB_NAME>
DB_USERNAME=<DB_USERNAME>
DB_PASSWORD="<DB_PASSWORD>"
MLFLOW_S3_ENDPOINT_URL=<MLFLOW_S3_ENDPOINT_URL>
AWS_BUCKET=<AWS_BUCKET>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

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
```
## External storage of metadata and artifacts

* [Backend Store (PostgreSQL)](https://mlflow.org/docs/latest/tracking.html#id77)
* [Artifact Store (S3)](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)


## Deployment with docker-compose
https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039


## Running Label Studio
label-studio -db postgresql