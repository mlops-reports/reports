# import tensorflow as tf
# from tensorflow import keras
import nltk
import simplemma
# import pandas as pd
import pathlib
import json
# import numpy as np
import re
import matplotlib.pyplot as plt

working_dir = pathlib.Path.cwd()

def find_and_replace_exact_word(input_word, replacements):
    """
    The function finds and replaces an exact word in a given input string with a replacement value.
    
    Args:
      input_word: The word that you want to find and replace in the text.
      replacements: A dictionary where the keys are the words to be replaced and the values are the
    words to replace them with.
    
    Returns:
      the value associated with the input_word in the replacements dictionary if it is an exact match.
    """
    for key, value in replacements.items():
        if input_word == key:
            return value
    return input_word

def sentence_cleaning_pipeline(sentences):
    """
    1. Remove punctuation
    2. Only lower letters
    3. Lemmatize
    4. Remove stop words
    5. Manual corrections
    """
    # tokenizer without punctuation
    regexp_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

    # get Turkish stopwords
    nltk.download("stopwords")
    turkish_stopwords = nltk.corpus.stopwords.words("turkish")

    # word corrections
    with open(
        working_dir.parent / "data" / "input" / "manuel_corrections.json", "r"
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
    """
    The function `find_longest_sentence_length` takes a list of sentences as input and returns the
    length of the longest sentence in terms of the number of words.
    
    Args:
      sentences: A list of sentences.
    
    Returns:
      the length of the longest sentence in the given list of sentences.
    """
    longest_length = max(len(sentence.split()) for sentence in sentences)
    return longest_length


def plot_graphs(history, string):
    """
    The function `plot_graphs` plots the training and validation values of a given metric over epochs.
    
    Args:
      history: The "history" parameter is the history object returned by the fit() method of a Keras
    model. It contains the training and validation metrics for each epoch.
      string: The "string" parameter is a string that represents the name of the metric or loss function
    that you want to plot. For example, if you want to plot the accuracy, you would pass "accuracy" as
    the value for the "string" parameter.
    """
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


def calculate_matching_ratio(list1, list2):
    """
    The function calculates the matching ratio between two lists by comparing the elements at
    corresponding indices.
    
    Args:
      list1: The first list of elements to compare.
      list2: The `list2` parameter is a list of values that will be compared to the values in `list1` to
    calculate the matching ratio.
    
    Returns:
      the matching ratio between two lists.
    """
    if len(list1) != len(list2):
        return "Lists are not of the same length"
    matches = sum(i == j for i, j in zip(list1, list2))
    return matches / len(list1)

def pattern_find_and_remove(text: str, pattern: str) -> tuple[str]:
    """
    The function `pattern_find_and_remove` takes a text and a pattern as input, finds the first
    occurrence of the pattern in the text, and returns a tuple containing the extracted pattern and the
    text with the pattern removed.
    
    Args:
      text (str): The `text` parameter is a string that represents the text in which we want to find and
    remove a specific pattern.
      pattern (str): The `pattern` parameter is a string that represents the pattern you want to find in
    the `text` parameter. It can be any valid regular expression pattern.
    
    Returns:
      The function `pattern_find_and_remove` returns a tuple containing two elements. The first element
    is the extracted pattern found in the text, and the second element is the text with the extracted
    pattern removed and stripped of leading and trailing whitespace.
    """
    try:
        extracted_pattern = re.findall(pattern, text)[0]
    except IndexError:
        extracted_pattern = ""

    return extracted_pattern, text.replace(extracted_pattern, "").strip()

def replace_patterns_from_text(
    text: str, patterns: list[str], replace_with: str = ""
) -> str:
    """
    The function `replace_patterns_from_text` takes a text string, a list of patterns, and an optional
    replacement string, and replaces all occurrences of the patterns in the text with the replacement
    string.
    
    Args:
      text (str): The `text` parameter is a string that represents the input text from which patterns
    will be replaced.
      patterns (list[str]): A list of patterns (strings) that you want to replace in the text.
      replace_with (str): The `replace_with` parameter is a string that specifies what each pattern
    should be replaced with in the `text` string.
    
    Returns:
      a modified version of the input text where all occurrences of the patterns in the list have been
    replaced with the specified replacement string.
    """
    for pattern in patterns:
        text = text.replace(pattern, replace_with)

    return text.strip()


def prepare_data_transformations(text) -> tuple[str]:
    """
    The function `prepare_data_transformations` takes in a text and performs various transformations on
    it, including replacing patterns, finding and removing specific patterns, and returning the
    transformed text along with extracted values for "rapor_tarihi" and "film_no".
    
    Args:
      text: The `text` parameter is a string that contains the text data to be processed.
    
    Returns:
      The function `prepare_data_transformations` returns a tuple containing three elements:
    """
    # regex patterns
    film_no_pattern = r"\b\d{6,}\b"
    rapor_tarihi_pattern= r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"

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