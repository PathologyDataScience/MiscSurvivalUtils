import pandas as pd
import numpy as np
from typing import Union, Iterable, Dict


def simplify_categorical_variables(df, prefixes_per_column):
    """Simplify categorical variables (details below).

    Say we have a categorical column 'Stage' for cancer stage, with values like
    'Stage IA', 'Stage IB', 'Stage IIAi', and so on. But we only care about
    the coarse stage (I, II, III, IV), then we could use this function as
    outlined in the example below.

    Parameters
    ----------
    df: pd.DataFrame
    prefixes_per_column: Dict[str,Iterable]
        Each dict index is the name of a column we want to code, while values
        are prefixes to be grouped together. IMPORTANT NOTE: the order matters!
        Make sure to provide the most specific strings first. For example,
        for stage, order should be 'StageIII` -> 'StageII' -> 'StageI'. In
        other words, if one prefix is a subset of another, it should come
        LAST.

    Returns
    -------

    Examples
    --------
        >>> df = simplify_categorical_variables(
        >>>     df, prefixes_per_column={"Stage": ["IV", "III", "II", "I"]}
        >>> )
    """

    def truncate(value, pfxs):
        if not isinstance(value, str):
            return value

        for pfx in pfxs:
            if value.startswith(pfx):
                return pfx

    for colname, prefixes in prefixes_per_column.items():
        df.loc[:, colname] = df.loc[:, colname].apply(
            lambda x: truncate(x, prefixes)
        )

    return df


def get_dummies_with_nan_preservation(
        data, categorical_columns=None, stringify=False
):
    """Converts df to dummy

    Arguments
    ----------
        data: pd.DataFrame
        categorical_columns: Iterable, list of categorical column names
        stringify: Boolean, convery all categorical values to strings?

    Returns
    -------
    pd.DataFrame
        pandas dataframe with dummy variables
    """

    def _stringify(x):
        if isinstance(x, float) and np.isfinite(x):
            x = int(x)
        return str(x)

    if categorical_columns is not None:
        non_categorical_columns = [
            c for c in data.columns if c not in categorical_columns
        ]
    else:
        categorical_columns = list(data.columns)
        non_categorical_columns = []

    # for dummification, categories should ideally be strings!
    if stringify:
        only_categoricals = data.loc[:, categorical_columns].applymap(
            lambda x: _stringify(x)
        )
    else:
        only_categoricals = data.loc[:, categorical_columns]

    # first non-categorical, then dummied categorical
    df = pd.concat(
        [
            data.loc[:, non_categorical_columns],
            pd.get_dummies(only_categoricals, dummy_na=True),
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

            keep = df.loc[:, [nan_colname]].iloc[:, 0] == 1
            nanidxs = list(df.loc[keep, :].index)

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
    df: pd.DataFrame
        dataframe containing data
    colnames: list
        names of columns to combine
    basestring: str
        string common to all variables being combined
    counter_basestrings: tuple
        strings NOT present in variable names

    Returns
    -------
    pd.DataFrame
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
