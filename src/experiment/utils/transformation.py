import nltk
import simplemma

# import pandas as pd
import pathlib
import json

# import numpy as np
import os
import re
import matplotlib.pyplot as plt

import openai


def get_project_root() -> pathlib.Path:
    """The function `get_project_root()` returns the path of the MLflow project root directory.

    Returns
    -------
      a `pathlib.Path` object representing the project root directory.

    """
    return pathlib.Path(os.getenv("MLFLOW_PROJECT_ROOT", None))


def find_and_replace_exact_word(input_word, replacements):
    """The function finds and replaces an exact word in a given input string with a replacement value.

    Parameters
    ----------
    input_word
      The input word is the word that you want to find and replace in a given text.
    replacements
      A dictionary where the keys are the words to be replaced and the values are the words to replace
    them with.

    Returns
    -------
      the value associated with the input_word in the replacements dictionary if it is found, otherwise
    it returns the input_word itself.

    """
    for key, value in replacements.items():
        if input_word == key:
            return value
    return input_word


def sentence_cleaning_pipeline(sentences):
    """
    1. Remove punctuation
    2. Only lower letters
    3. skip_pre_lemmatasation
    4. Lemmatize
    5. Remove stop words
    6. fix_post_lemmatasation
    """
    # tokenizer without punctuation
    regexp_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

    # get Turkish stopwords
    nltk.download("stopwords", quiet=True)
    turkish_stopwords = nltk.corpus.stopwords.words("turkish")

    # word corrections
    with open(
        get_project_root() / "data" / "input" / "manuel_corrections.json", "r"
    ) as file:
        manuel_corrections = json.load(file)

    return [
        " ".join(
            find_and_replace_exact_word(
                simplemma.lemmatize(word, lang="tr").lower(),
                manuel_corrections["fix_post_lemmatasation"],
            )
            for word in regexp_tokenizer.tokenize(sentence)
            if word
            not in turkish_stopwords + manuel_corrections["skip_pre_lemmatasation"]
        )
        for sentence in sentences
    ]


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


def plot_graphs(history, string):
    """The function `plot_graphs` plots the training and validation values of a given metric over epochs.

    Parameters
    ----------
    history
      The "history" parameter is the history object returned by the fit() method of a Keras model. It
    contains the training and validation metrics for each epoch.
    string
      The "string" parameter is a string that represents the name of the metric or loss function that you
    want to plot. For example, if you want to plot the accuracy, you would pass "accuracy" as the value
    for the "string" parameter.

    """
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


def pattern_find_and_remove(text: str, pattern: str) -> tuple[str]:
    """The function `pattern_find_and_remove` takes a text and a pattern as input, finds the first
    occurrence of the pattern in the text, and returns a tuple containing the extracted pattern and the
    text with the pattern removed.

    Parameters
    ----------
    text : str
      The `text` parameter is a string that represents the text in which we want to find and remove a
    specific pattern.
    pattern : str
      The `pattern` parameter is a string that represents the pattern you want to find in the `text`
    parameter.

    Returns
    -------
      The function `pattern_find_and_remove` returns a tuple containing two elements.

    """
    try:
        extracted_pattern = re.findall(pattern, text)[0]
    except IndexError:
        extracted_pattern = ""

    return extracted_pattern, text.replace(extracted_pattern, "").strip()


def replace_patterns_from_text(
    text: str, patterns: list[str], replace_with: str = ""
) -> str:
    """The function `replace_patterns_from_text` replaces multiple patterns in a given text with a
    specified replacement string.

    Parameters
    ----------
    text : str
      The `text` parameter is a string that represents the input text from which patterns will be
    replaced.
    patterns : list[str]
      A list of patterns (strings) that you want to replace in the text.
    replace_with : str
      The `replace_with` parameter is a string that specifies what each pattern should be replaced with
    in the `text` string. By default, if no value is provided for `replace_with`, it will be an empty
    string.

    Returns
    -------
      a modified version of the input text where all occurrences of the patterns in the list have been
    replaced with the specified replacement string. The returned text is also stripped of leading and
    trailing whitespace.

    """
    for pattern in patterns:
        text = text.replace(pattern, replace_with)

    return text.strip()


def translate_report(report: str) -> str:
    '''The `translate_report` function uses OpenAI's GPT-3.5-turbo model to translate a given report from
    an unspecified language into English.
    
    Parameters
    ----------
    report : str
      The `report` parameter is a string that represents the text that needs to be translated into
    English.
    
    Returns
    -------
      The function `translate_report` is returning the translated report as a string.
    
    '''
    openai.api_key = os.getenv("OPEN_AI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate this text into English: {report}"}
        ],
    )

    translated_report = response.choices[0].message.content

    if translate_report == "":
        raise ValueError("Translation failed.")

    return translated_report

def prepare_data_transformations(text) -> tuple[str]:
    """The function `prepare_data_transformations` takes in a text and performs various transformations on
    it, including replacing patterns, finding and removing specific patterns, and returning the
    transformed text along with extracted values for "rapor_tarihi" and "film_no".

    Parameters
    ----------
    text
      The `text` parameter is a string that contains the text data to be processed.

    Returns
    -------
      The function `prepare_data_transformations` returns a tuple containing three elements:

    """
    # regex patterns
    film_no_pattern = r"\b\d{6,}\b"
    rapor_tarihi_pattern = r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"

    text = replace_patterns_from_text(
        text,
        [
            "RAPOR TARİHİ",
            "FİLM NO",
            "TETKİK NO",
            "ÇEKİM TARİHİ",
            "ÇEKİM  TARİHİ",
            "TETKİK TARİHİ",
            "Tetkik no",
            " :",
            ": ",
            " : ",
        ],
    )
    text = replace_patterns_from_text(
        text, ["RAPOR TARİHİ", "FİLM NO", "TETKİK NO", " :", ": ", " : "]
    )
    text = replace_patterns_from_text(text, ["\n", "*"], " ")
    text = replace_patterns_from_text(text, [":", ";"], " ")

    film_no, text = pattern_find_and_remove(
        text,
        film_no_pattern,
    )
    rapor_tarihi, text = pattern_find_and_remove(text, rapor_tarihi_pattern)

    return text.strip(), rapor_tarihi, film_no
