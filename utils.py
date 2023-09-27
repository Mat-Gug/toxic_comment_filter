import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

def show_barplot(x, y, title=None, xlabel=None, ylabel=None, xtickrotation=True):
    sns.set(font_scale=2)
    plt.figure(figsize=(12,7))
    ax = sns.barplot(x=x, y=y)
    if title is not None:
        plt.title(title, fontsize=24)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=18)
    rects = ax.patches
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5,
                int(height), fontsize=18,
                ha='center', va='bottom')
    if xtickrotation:
        plt.xticks(rotation=45, ha='center', fontsize=16)
    else:
        plt.xticks(ha='center', fontsize=16)
    plt.yticks(fontsize=14)
    plt.show()

def stemming(sentence):
    """
    Apply stemming to a sentence.
    This function uses the Snowball Stemmer for English to perform stemming
    on each word in a given sentence. Stemming reduces words to their
    root or base form.

    Parameters:
    -----------
    sentence : str
      The input sentence to be stemmed.

    Returns:
    --------
    str
      The stemmed sentence where each word has been reduced to its root form.

    Example:
    --------
    >>> sentence = "Running quickly in the park"
    >>> stemming(sentence)
    'run quick in the park'
    """
    stemmer = SnowballStemmer("english")
    stem_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence

def clean_text(text):
    """
    Clean and preprocess text data.

    This function takes a text input and performs the following cleaning steps:
    1. Converts text to lowercase.
    2. Removes words containing only one letter or one letter followed/preceded by
      one or more punctuation characters, plus tabs and line breaks,
      minus the ' character.
    3. Removes common English stopwords.

    Parameters:
    ----------
    text : str
      The input text to be cleaned.

    Returns:
    -------
    str
      The cleaned and preprocessed text, returned as a
      space-separated string of words.
    """
    swords = set(stopwords.words("english"))
    re_stop_words = re.compile(r"\b(" + "|".join(swords) + r")\b", re.I)
    regex = r'^[a-z]{1}[!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n]*$|'+\
          r'^[!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n]*[a-z]{1}$'
    text = text.lower()
    text = " ".join(word for word in text.split() if not re.match(regex, word))
    text = " ".join(word for word in text.split() if not re.search(re_stop_words, word))
    text = stemming(text)
    return text

def text_preprocessing(data, column_name="comment_text"):
    """
    Preprocess text data in a DataFrame column.

    This function takes a pandas DataFrame and preprocesses the text data
    in a specified column by applying the `clean_text` function to each element.

    Parameters:
    ----------
    data : pandas.DataFrame
      The DataFrame containing the text data to be preprocessed.
    column_name : str, optional
      The name of the column containing the text data to be preprocessed.
      Default is "comment_text".

    Returns:
    -------
    pandas.DataFrame
      The DataFrame with the specified column's text data
      preprocessed using the `clean_text` function.
    """
    data[column_name] = data[column_name].apply(clean_text)
    return data

def train_test_val_split(X, y, train_size, val_size, random_state=None):
    """
    Split the input data into training, validation, and test sets.
    This function takes the input data `X` and corresponding labels `y` and
    splits them into training, validation, and test sets based on the specified
    ratios.

    Parameters:
    -----------
    X : array-like
      The input data to be split.
    y : array-like
      The corresponding labels for the input data.
    train_size : float
      The ratio of the data to include in the training set.
    val_size : float
      The ratio of the data to include in the validation set.
    random_state : int or None, optional
      Seed for random number generation. If None, no seed is used.

    Returns:
    --------
    tuple
      A tuple containing the split data in the following order:
      (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_size_2 = val_size/train_size
    X_train, X_test, y_train, y_test = \
      train_test_split(X, y, train_size=train_size, random_state=random_state)
    X_train, X_val, y_train, y_val = \
      train_test_split(X_train, y_train, test_size=test_size_2, random_state=random_state)
    return (X_train, X_val, X_test, y_train, y_val, y_test)

def plot_learning_curves(hist, exp_name):
    plt.figure(figsize=(15,8))
    for subplot, curve in enumerate(['loss','hamming_loss']):
        plt.subplot(1,2,subplot+1)
        plt.plot(hist.history[curve],label='training')
        plt.plot(hist.history['val_'+curve],label='validation')
        plt.legend()
        plt.title(exp_name+':'+curve)
    plt.tight_layout()
    plt.show()

def plot_mcms(model, X, y_true, threshold=.5, labels=None):
    y_pred_proba = model.predict(X)
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    if labels is not None:
        labels = labels
    else:
        labels = [f"label_{i}" for i in range(1,y_true.shape[1]+1)]
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    for i, ax in enumerate(axes.flat):
        if i<len(labels):
            tn, fp, fn, tp = mcm[i].ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            df_cm = pd.DataFrame(
                mcm[i],
                index=["Negative", "Positive"],
                columns=["Negative", "Positive"]
            )
            sns.set(font_scale=1.4)
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 20})
            ax.set_title(f"{labels[i]}: Precision={precision:.3f}, Recall={recall:.3f}", fontsize=18)
            ax.set_xlabel("Predicted", fontsize=16)
            ax.set_ylabel("Actual", fontsize=16)
        else:
            ax.axis('off')
    plt.tight_layout()
    # plt.savefig("nome_del_file.png")
    plt.show()

def get_sequences(X_train, X_val, X_test, num_words=1000):
  tokenizer = Tokenizer(num_words = num_words)
  tokenizer.fit_on_texts(X_train)
  train_sequences= tokenizer.texts_to_sequences(X_train)
  val_sequences= tokenizer.texts_to_sequences(X_val)
  test_sequences= tokenizer.texts_to_sequences(X_test)
  vocab_size = max(
    [index for sequence in train_sequences for index in sequence]
  )+1
  maxlen = len(max(train_sequences,key=len))
  padded_train_sequences = pad_sequences(train_sequences, maxlen=maxlen)
  padded_val_sequences = pad_sequences(val_sequences, maxlen=maxlen)
  padded_test_sequences = pad_sequences(test_sequences, maxlen=maxlen)
  return (padded_train_sequences, 
          padded_val_sequences, 
          padded_test_sequences,
          vocab_size,
          maxlen)