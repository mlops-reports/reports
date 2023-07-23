# mlflow

## Description
This is a template for the ml experimentation use cases

## Creating a Virtual Environment
```
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage on local machine
create `.env` file in working directory containing required access tokens
```
DB_HOST=<DB_HOST>
DB_NAME=<DB_NAME>
DB_USERNAME=<DB_USERNAME>
DB_PASSWORD="<DB_PASSWORD>"
MLFLOW_S3_ENDPOINT_URL=<MLFLOW_S3_ENDPOINT_URL>
AWS_BUCKET=<AWS_BUCKET>
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```
## External storage of metadata and artifacts

* [Backend Store (PostgreSQL)](https://mlflow.org/docs/latest/tracking.html#id77)
* [Artifact Store (S3)](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)


## Deployment with docker-compose
https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039