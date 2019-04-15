import pandas as pd
import sys


def read_data(data_path):
    """
    read the csv file as a data frame
    :param data_path: the path of the csv fie we want to work on (only the name of the csv file)
    :return: a pandas data frame
    """
    return pd.read_csv(data_path, error_bad_lines=False, encoding='latin-1')


def delete_redundant_column(col_name, data_frame):
    """
    Delete useless variables on the data frame
    :param col_name: the name of the colúmn we want to drop
    :param data_frame: the data frame concerned
    :return:
    """
    return data_frame.drop(columns=col_name)


file_path = sys.argv[1]
page_df = read_data(file_path)
col_to_drop = "referringpageinstanceid"
page_df_cleaned = delete_redundant_column(col_to_drop, page_df)
input_name = file_path.split(".")[0]
out_put_name = input_name + "_cleaned.csv"
page_df_cleaned.to_csv(out_put_name)

