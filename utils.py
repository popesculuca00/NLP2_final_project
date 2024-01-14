import re
import pandas as pd

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

def read_process_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df.columns = df.columns.map(rename_columns)

    string_columns = ["text_id", "text", "selected_text", "country", "time_of_tweet", "age_of_user"]
    int_columns = ["population", "land_area", "density"]
    for str_col in string_columns:
        if str_col in df.columns: 
            df[str_col] = df[str_col].astype('str')

    for int_col in int_columns:
        if int_col in df.columns: 
            df[int_col] = pd.to_numeric(df[int_col])

    return df