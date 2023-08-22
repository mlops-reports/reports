from experiment.api import mlflow as mlflow_api
from experiment.utils import transformation

# import pandas as pd
import json
import sys

import mlflow as mlflow_lib

# import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import tokenizer_from_json


def main():
    mlflow = mlflow_api.MLFlow()

    # labels
    results: dict = {
        0: "Emergency",
        1: "Normal",
        2: "Non Emergency [Doctor]",
        3: "Non Emergency [No Doctor]",
    }

    argument = sys.argv[1]

    # get an example report
    # example_sentence = """
    #     Sağ maksiller sinüste retansiyon kisti izlenmiştir.    
    # """
    sentences = [argument]
    
    mlflow.logger.info(f"Input sentence: {sentences}")
    
    # clean_labels = [0]
    # print(f"gold standard: {clean_labels}")

    # get model config
    model_config_path = (
        transformation.get_project_root()
        / "data"
        / "input"
        / "model_config"
        / "nlp_experiment.json"
    )
    model_config = mlflow.get_model_config(model_config_path)

    # get the best run & generate the tokenizer & format the input sentences
    best_run = mlflow.get_best_run_by_metric("NLP Experiments", "val_accuracy")
    clean_sentences = transformation.sentence_cleaning_pipeline(sentences)
    tokenizer_artifact: json = mlflow_lib.artifacts.load_dict(
        artifact_uri=f"{best_run.artifact_uri}/data/tokenizer.json"
    )
    tokenizer: Tokenizer = tokenizer_from_json(json.dumps(tokenizer_artifact))
    tokenizer.fit_on_texts(clean_sentences)
    clean_sequences = tokenizer.texts_to_sequences(clean_sentences)
    clean_sequences_padded = pad_sequences(
        clean_sequences,
        maxlen=model_config["max_input_length"],
        padding=model_config["padding_type"],
        truncating=model_config["trunc_type"],
    )
    clean_sequences_request = clean_sequences_padded.tolist()
  
    predictions = mlflow.get_predictions(clean_sequences_request)
    prediction_texts = [
        f"{index}: {results[prediction]}"
        for index, prediction in enumerate(predictions)
    ]

    mlflow.logger.info(prediction_texts)


if __name__ == "__main__":
    main()
