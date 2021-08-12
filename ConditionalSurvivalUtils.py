# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:20:16 2018
@author: computer
"""
import pandas as pd
import numpy as np
from typing import Union
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation

import os
from os.path import join as opj
import sys

BASEPATH = opj(os.path.expanduser("~"), "Desktop", "MiscSurvivalUtils")
sys.path.insert(0, BASEPATH)
from DataCodingUtils import prep_data_for_conditional_survival


def conditional_cox(
    source_table,
    source_table_survival_column,
    source_table_event_column,
    time_passed,
    verbose=True,
):
    """
    Calculates a cox regression conditional on the patients having survived
    to a certain point in time.

    Arguments
    ----------
    source_table: pd.DataFrame
        pandas DF containing time to event, event, and variables as columns
        NOTE: all variables should be either continuous or dummy variables
    source_table_survival_column: str
        name of time to event column
    source_table_event_column: str
        name of "event" column -- 1 means dead, 0 means censored
    time_passed: Union[int,float]
        time that has passed since start of study period
    verbose: bool

    Returns
    -------
    pd.DataFrame
        cph.summary pandas dataframe
    """

    cph = CoxPHFitter()

    # Prep for conditional survival
    source_table_slice = prep_data_for_conditional_survival(
        source_table=source_table,
        source_table_survival_column=source_table_survival_column,
        time_passed=time_passed,
    )
    # now fit model
    cph.fit(
        source_table_slice,
        duration_col=source_table_survival_column,
        event_col=source_table_event_column,
        show_progress=verbose,
    )
    return cph


def conditional_cox_cross_validation(
    source_table,
    source_table_survival_column,
    source_table_event_column,
    time_passed,
    number_of_folds,
):
    """
    Does cross-validated cox regression using conditional on the patients
    having survived to a certain point in time.

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
    dict
        Various relevant attributes of model
    """
    cph = CoxPHFitter()
    # Prep for conditional survival
    source_table_slice = prep_data_for_conditional_survival(
        source_table,
        source_table_survival_column=source_table_survival_column,
        time_passed=time_passed,
    )
    # do cross validation
    scores = k_fold_cross_validation(
        fitters=cph,
        df=source_table_slice,
        duration_col=source_table_survival_column,
        event_col=source_table_event_column,
        k=number_of_folds,
        scoring_method="concordance_index",
    )
    to_return = {"c_index_folds": scores, "N": source_table_slice.shape[0]}

    return to_return


def get_conditional_survival(
    source_table, source_table_label, time_passed, time_to_survive
):
    """conditional survival model"""
    CS = (
        source_table.loc[time_passed + time_to_survive, source_table_label]
        / source_table.loc[time_passed, source_table_label].copy()
    )
    CS = CS * 100
    return CS


def get_univariable_model(
    data_frame, outcome_str, time_passed, durationstr="SURVIVAL", K=5
):
    """
    Gets univariable cox regression model, including model accuracy
    on testing set (cross-validation), training set, and final
    model coefficients.

    Arguments
    ---------
    data_frame: pd.DataFrame
        a pandas dataframe, contains outcomes are columns as well
    outcome_str: str
        eg 'OS' for overall survival -- column name of event indicator
    durationstr: str
        duration column name
    K: int
        no of folds
    time_passed: int
        numerical, represents the number of years patients already lived

    Returns
    --------
    pd.DataFrame
        dataframe of results
    """
    print(
        "Getting Univariable model for",
        outcome_str,
        "after",
        time_passed,
        "year(s) have passed",
    )
    varnames = list(data_frame.columns)
    varnames.remove(durationstr)
    varnames.remove(outcome_str)

    model_df = pd.DataFrame(index=varnames)
    coeffs_df = pd.DataFrame()

    for varname in varnames:
        print("  univariable model for", varname)
        source_table_slice = data_frame.loc[:, [varname] + [durationstr, outcome_str]]

        # testing accuracy (cross validation)
        cv_results = conditional_cox_cross_validation(
            source_table=source_table_slice,
            source_table_survival_column=durationstr,
            source_table_event_column=outcome_str,
            time_passed=time_passed,
            number_of_folds=K,
        )
        model_df.loc[varname, "N"] = cv_results["N"]
        for k in range(1, K + 1):
            model_df.loc[varname, "cindex_K-%d" % k] = cv_results["c_index_folds"][
                k - 1
            ]

        # model coefficients and training model fit
        cph = conditional_cox(
            source_table=source_table_slice,
            source_table_survival_column=durationstr,
            source_table_event_column=outcome_str,
            time_passed=time_passed,
            verbose=False,
        )
        cph_summary = cph.summary.copy()
        cph_summary.loc[:, "exp(lower 0.95)"] = np.exp(cph_summary.loc[:, "lower 0.95"])
        cph_summary.loc[:, "exp(upper 0.95)"] = np.exp(cph_summary.loc[:, "upper 0.95"])
        cph_summary.loc[:, "cindex_training"] = cph.score_
        coeffs_df = pd.concat((coeffs_df, cph_summary), axis=0)

    # concat accuracy and coefficients for nice presentation
    model_df = pd.concat((model_df, coeffs_df), axis=1)

    print("done")

    return model_df


def get_multivariable_model(
    data_frame, outcome_str, time_passed, durationstr="SURVIVAL", K=5
):
    """
    Gets multivariable cox regression model, including model accuracy
    on testing set (cross-validation), training set, and final
    model coefficients.

    Arguments
    ---------
    data_frame: pd.DataFrame
        a pandas dataframe, contains outcomes are columns as well
    outcome_str: str
        eg 'OS' for overall survival -- column name of event indicator
    durationstr: str
        duration column name
    K: int
        no of folds
    time_passed: int
        represents the number of years patients already lived

    Returns
    -------
    pd.DataFrame
        the results of the cox model
    pd.DataFrame
        the C-index scoresfor each fold
    """
    print(
        "Getting multivariable model for",
        outcome_str,
        "after",
        time_passed,
        "year(s) have passed",
    )

    source_table_slice = data_frame.copy()

    if K > 0:
        # testing accuracy (cross validation)
        cv_results = conditional_cox_cross_validation(
            source_table=source_table_slice,
            source_table_survival_column=durationstr,
            source_table_event_column=outcome_str,
            time_passed=time_passed,
            number_of_folds=K,
        )
        cv_results = pd.DataFrame.from_dict(cv_results)
    else:
        cv_results = None

    # model coefficients and training model fit
    cph = conditional_cox(
        source_table=source_table_slice,
        source_table_survival_column=durationstr,
        source_table_event_column=outcome_str,
        time_passed=time_passed,
        verbose=False,
    )

    cph_summary = cph.summary.copy()
    cph_summary.loc[:, "cindex_training"] = cph.concordance_index_

    print("done")

    return cph_summary, cv_results
