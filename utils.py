import re, string
import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

sw = stopwords.words('english')


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatization(data, lemmatizer):

    tokens = word_tokenize(data)
    tags = [nltk.tag.pos_tag([token]) for token in tokens]
    data = [lemmatizer.lemmatize(pair[0][0], pos=get_wordnet_pos(pair[0][1])) for pair in tags]
    return " ".join(data)


def get_contractions(path):

    contr_char = chr(39) # `
    text_char  = chr(96) # '

    df_contractions = pd.read_csv(path)
    for col_name in df_contractions.columns:
        df_contractions[col_name] = df_contractions[col_name].replace({contr_char: text_char}, regex=True).str.lower()
    df_contractions.set_index('Contraction', inplace=True)
    dict_contractions = df_contractions.to_dict()['Meaning']

    return dict_contractions


def rename_columns(col_name):

    # Removing the unnecessary additional info in column name 
    new_name = re.findall("(.*)\s(?:-|\(|$)", str(col_name))
    if len(new_name):
        col_name = new_name[0]

    # For the textId column 
    col_name = re.sub("([^\s])([A-Z])", r"\1_\2", col_name)
    # For all other columns
    col_name = re.sub("( [a-zA-Z])", r"_\1", col_name).lower().replace(" ", "")

    return col_name


def data_cleaning(data, dict_contractions=None):

    data = data.lower().strip()

    # replace contractions (i`ve -> i have, couldn`t -> could not etc.)
    if dict_contractions is not None:
        data_split = data.split(" ")
        for idx_word in range(len(data_split)):
            if data_split[idx_word].strip() in dict_contractions.keys():
                data_split[idx_word] = dict_contractions[data_split[idx_word].strip()]
        data = " ".join(data_split)

    # removing punctuation and digits
    data = re.sub(r"[^a-z]+", " ", data, flags=re.UNICODE)

    clean_text = []
    for word in data.split(" "):
      if word not in sw and word not in ['', 'http', 'lol', 'www', 'com'] and len(word) >= 3:
        clean_text.append(word)

    return " ".join(clean_text)


def preprocess_text(df, column_name, lemm=False, dict_contractions=None):

    lemmatizer = WordNetLemmatizer()
    df[column_name + "_preprocessed"] = df[column_name].apply(lambda text: data_cleaning(lemmatization(text, lemmatizer),
                                                                                         dict_contractions) \
                                                              if lemm else data_cleaning(text, dict_contractions))
    return df


def remove_nan_rows(df):
    return df.dropna(subset=['text_id'])

def read_process_data(path, preprocess_text_flg=False, lemm=False, contractions_path=None, remove_nan_flag=False):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = df.columns.map(rename_columns)

    if remove_nan_flag:
        df = remove_nan_rows(df)

    string_columns = ["text_id", "text", "selected_text", "country", "time_of_tweet", "age_of_user"]
    int_columns = ["population", "land_area", "density"]
    for str_col in string_columns:
        if str_col in df.columns: 
            df[str_col] = df[str_col].astype('str')

    for int_col in int_columns:
        if int_col in df.columns: 
            df[int_col] = pd.to_numeric(df[int_col])

    # create a new column text_preprocessed for the preprocessed text
    if preprocess_text_flg:
        dict_contractions = None
        if contractions_path is not None:
            dict_contractions = get_contractions(contractions_path)
        df = preprocess_text(df, "text", lemm, dict_contractions)
        df = df[df["text_preprocessed"] != '']

    return df
