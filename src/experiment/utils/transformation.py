# import nltk
# import simplemma

# import numpy as np
import os

# import pandas as pd
import pathlib

import openai

# import json
from cryptography.fernet import Fernet

# import re
# import matplotlib.pyplot as plt


def get_project_root() -> pathlib.Path:
    """The function `get_project_root()` returns the path of the MLflow project root directory.

    Returns
    -------
      a `pathlib.Path` object representing the project root directory.

    """
    return pathlib.Path(os.getenv("MLFLOW_PROJECT_ROOT", None))


def hash_value(text: str) -> str:
    """The function `hash_value` takes a string as input, encrypts it using a secret key, and returns the
    encrypted value as a string.

    Parameters
    ----------
    text : str
      The `text` parameter is a string that represents the text that you want to hash.

    Returns
    -------
      a hashed value of the input text.

    """
    secret_key = os.getenv("FERNET_KEY")
    cipher_suite = Fernet(secret_key)
    return cipher_suite.encrypt(text.encode("utf-8")).decode("utf-8")


def unhash_value(hashed_text: str) -> str:
    """The function `unhash_value` takes a hashed text as input, decrypts it using a secret key, and
    returns the original text.

    Parameters
    ----------
    hashed_text : str
      The `hashed_text` parameter is a string that represents the encrypted value that needs to be
    decrypted.

    Returns
    -------
      the decrypted value of the hashed text.

    """
    secret_key = os.getenv("FERNET_KEY")
    cipher_suite = Fernet(secret_key)
    return cipher_suite.decrypt(hashed_text.encode("utf-8")).decode("utf-8")


def find_longest_sentence_length(sentences):
    """The function `find_longest_sentence_length` takes a list of sentences as input and returns the
    length of the longest sentence in terms of the number of words.

    Parameters
    ----------
    sentences
      A list of sentences.

    Returns
    -------
      the length of the longest sentence in the given list of sentences.

    """
    longest_length = max(len(sentence.split()) for sentence in sentences)
    return longest_length


def prompt_report(
    report: str,
    prompt: str = "Remove dates and film numbers, translate to English, and remove spaces between sentences",
) -> str:
    """The `prompt_report` function takes a report in a specified language and uses OpenAI's
    GPT-3.5-turbo model to translate it into English.

    Parameters
    ----------
    report : str
      The `report` parameter is a string that represents the text that you want to translate into
    English. It can be any text in any language.
    prompt : str, optional
      The `prompt` parameter is a string that represents the instruction or question given to the model.
    In this case, the default prompt is "Translate this text into English".

    Returns
    -------
      The function `prompt_report` returns the translated report as a string.

    """
    openai.api_key = os.getenv("OPEN_AI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompt}: {report}"}],
    )

    prompted_report = response.choices[0].message.content

    if prompted_report == "":
        raise ValueError("Prompting failed.")

    return prompted_report
