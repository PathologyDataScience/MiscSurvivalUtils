import pandas as pd
import numpy as np
from typing import Union, Iterable


def simplify_categorical_variables(df, relevant_prefixes_per_column):
    """Simplify categorical variables (details below).

    Say we have a categorical variable for cancer stage. The values can be
    'Stage IA', 'Stage IB', 'Stage IIAi', and so on. But we only care about
    the

    Parameters
    ----------
    df
    relevant_prefixes_per_column

    Returns
    -------

    """
    pass


def get_dummies_with_nan_preservation(df, categorical_columns=None):
    """Converts df to dummy
    
    Arguments
    ----------
        df: pd.DataFrame
        categorical_columns: Iterable
        verbose: bool

    Returns
    -------
    pd.DataFrame
        pandas dataframe with dummy variables
    """
    if categorical_columns is not None:
        non_categorical_columns = [
            c for c in df.columns if c not in categorical_columns
        ]
    else:
        categorical_columns = list(df.columns)
        non_categorical_columns = []

    # first non-categorical, then dummied categorical
    df = pd.concat(
        [
            df.loc[:, non_categorical_columns],
            pd.get_dummies(df.loc[:, categorical_columns], dummy_na=True),
        ],
        axis=1
    )

    for colname in categorical_columns:

        # get corresponding dummy column names
        # eg GRADE_Moderately differentiated; Grade II', GRADE_Well, GRADE_nan
        dummy_colnames = [
            j
            for j in df.columns
            if ((j.startswith(colname + "_")) and not (j.endswith("_nan")))
        ]

        # get nan indices
        nan_colname = colname + "_nan"

        if nan_colname in df.columns:

            nanidxs = list(df.loc[df[nan_colname] == 1, :].index)

            # for every dummy column, make sure the original
            # nan values are preserved
            for dummy_colname in dummy_colnames:
                df.loc[nanidxs, dummy_colname] = np.nan

            # remove nan column
            df.drop(nan_colname, axis=1, inplace=True)

    return df


def prep_data_for_conditional_survival(
    source_table, source_table_survival_column, time_passed
):
    """
    Calculates a cox regression conditional on the patients having survived to
    a certain point in time.

    Arguments
    ---------
    source_table: pd.DataFrame
        pandas DF containing time to event, event, and variables as columns
        NOTE: all variables should be either continuous or dummy variables
    source_table_survival_column: str
        name of time to event column
    time_passed: Union[int,float]
        time that has passed since start of study period

    Returns
    -------
        a slice from the dataframe (copy)
    """

    # isolate patients that remain at-risk up to this point (survived so far)
    # NOTE: we use .copy() to make sure we don't modify the original dataframe
    source_table_slice = source_table.loc[
        source_table[source_table_survival_column] >= time_passed, :
    ].copy()
    # reset time to zero
    source_table_slice.loc[:, source_table_survival_column] = (
        source_table_slice.loc[:, source_table_survival_column] - time_passed
    )
    # clean up
    source_table_slice = source_table_slice.dropna()

    return source_table_slice


def combine_vars(
    df,
    colnames,
    basestring,
    counter_basestrings,
    newcolname=None,
    operator_type="OR",
    drop=True,
):
    """
    Combine binary variables.

    Arguments
    ---------
    df: DataFrame
        dataframe containing data
    colnames: list
        names of columns to combine
    basestring: str
        string common to all variables being combined
    counter_basestrings: tuple
        strings NOT present in variable names

    Returns
    -------
    DataFrame
        modified dataframe
    """
    var_names = []
    var_idxs = []
    for i, j in enumerate(colnames):

        counterstring_is_present = [cs in j for cs in counter_basestrings]

        if (basestring in j) and (True not in counterstring_is_present):
            var_names.append(j)
            var_idxs.append(i)
    new_var = df.values[:, var_idxs]

    if operator_type == "OR":
        new_var = 0 + (np.sum(new_var, axis=1) > 0)
    elif operator_type == "AND":
        new_var = 0 + (np.sum(new_var, axis=1) == new_var.shape[1])
    else:
        raise ValueError("Unknown operator_type")

    if drop:
        for varname in var_names:
            df = df.drop(varname, axis=1)

    if newcolname is not None:
        df[newcolname] = new_var
    else:
        df[basestring] = new_var

    return df
