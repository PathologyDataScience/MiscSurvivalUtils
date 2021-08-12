import pandas as pd
import numpy as np
from typing import Union


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
    source_table_event_column: str
        name of "event" column -- 1 means dead, 0 means censored
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


def get_dummies_with_nan_preservation(pd_df, verbose=True):
    """Converts df to dummy
    Arguments
    ----------
        pd_df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
        pandas dataframe with dummy variables
    """

    # get column names in original DF
    colnames_original = list(pd_df.columns)

    # convert to dummies, making sure to also note which values are NA
    pd_df = pd.get_dummies(pd_df, dummy_na=True)

    for colname in colnames_original:

        # get corresponding cummy column names
        # eg GRADE_Moderately differentiated; Grade II', GRADE_Well, GRADE_nan
        dummy_colnames = [
            j
            for j in pd_df.columns
            if ((j.startswith(colname + "_")) and not (j.endswith("_nan")))
        ]

        # get nan indices
        nan_colname = colname + "_nan"

        if nan_colname in pd_df.columns:

            if verbose:
                print("Preserving nans for ", colname)

            nanidxs = list(pd_df.loc[pd_df[nan_colname] == 1, :].index)

            # for every dummy column, make sure the original
            # nan values are preserved
            for dummy_colname in dummy_colnames:
                pd_df.loc[nanidxs, dummy_colname] = np.nan

            # remove nan column
            pd_df.drop(nan_colname, axis=1, inplace=True)

    return pd_df
