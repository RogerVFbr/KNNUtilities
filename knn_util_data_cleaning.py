import pandas as pd
from pandas.api.types import is_numeric_dtype, is_float_dtype


class KNNUtilDataCleaning:

    def __init__(self):
        pd.set_option('display.width', 320)
        pd.set_option('display.max_columns', 100)

    @staticmethod
    def prepare(df, print_log = False):
        
        for n, c in df.items():

            change_detected = False

            # Convert NaN on string columns to 'Other'.
            if not is_numeric_dtype(c) and c.isnull().values.any():
                    replace_by = 'Other'
                    df[n].fillna(replace_by, inplace=True)
                    change_detected = True
                    if print_log:
                        print(f"'{n}' ({c.dtype}) column's NaN/Null become '{replace_by}': "
                              f"({c.nunique()}) - {c.unique()}")

            # Apply mean to missing values on float columns.
            if is_float_dtype(c) and c.isnull().values.any():
                    df[f"{n}_isnull"] = df[n].isna()
                    mean_value = df[n].mean(skipna=True)
                    df[n].fillna(mean_value, inplace=True)
                    change_detected = True
                    if print_log:
                        print(f"'{n}' ({c.dtype}) column's NaN/Null become mean '{mean_value}': "
                              f"MIN - {c.min()} | MAX - {c.max()}")

            # Convert strings to categories.
            if not is_numeric_dtype(c):
                df[n] = df[n].astype('category').cat.as_ordered()
                change_detected = True
                new_values = df[n]
                if print_log:
                    print(f"'{n}' ({new_values.dtype}) column's categories became categorized "
                          f"({new_values.nunique()} cats)")

            # Convert categorical column to corresponding int codes.
            if not is_numeric_dtype(c):
                df[n] = df[n].cat.codes
                change_detected = True
                new_values = df[n]
                if print_log:
                    print(f"'{n}' ({new_values.dtype}) column's strings became ints "
                          f"({new_values.nunique()}) - {new_values.unique()}")

            if change_detected and print_log:
                print('.')

        return df


if __name__ == "__main__":

    train = pd.read_csv("data_sets/kaggle_house/train.csv")
    result = KNNUtilDataCleaning().prepare(train, print_log=True)
    print(result.head())