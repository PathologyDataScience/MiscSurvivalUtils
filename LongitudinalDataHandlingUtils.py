# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 04:01:29 2018
@author: mohamed
"""

from pandas import DataFrame as df, concat
import numpy as np


def make_wide(
    dataset_long, id_col_name="id", time_col_name="time", time_independent_covars=None
):
    """
    Converts a long dataset format to a wide format.
    Arguments
    ---------
        dataset_long: DataFrame
            table ing long format
        id_col_name: str
            name of column corresponding to cluster id
        time_col_name: str
            name of column corresponding to time indicator
        time_independent_covars: list, optional
            time-independent covariates
    Returns
    -------
    DataFrame
        pandas dataframe in wide data format
    """
    time_independent_covars = time_independent_covars or []
    # find unique ids and times
    ids_unique = list(np.unique(dataset_long["id"]))
    times_unique = list(np.unique(dataset_long["time"]))

    # Get wide colnames
    #
    time_dependent = list(dataset_long.columns)

    time_independent = [id_col_name, time_col_name]
    time_independent.extend(time_independent_covars)
    for toremove in time_independent:
        time_dependent.remove(toremove)

    # time independent covars go first, then time dependent
    colnames = time_independent.copy()
    colnames.remove(time_col_name)  # no more need for time column
    for t in times_unique:
        colnames.extend([j + "_t={}".format(t) for j in time_dependent])

    # Initialize result df
    dataset_wide = df(columns=colnames)

    # Go through individual clusters
    dfrow = -1
    for sid in ids_unique:
        dfrow += 1
        print("Cluster {} of {}".format(dfrow, len(ids_unique) - 1))
        cluster = dataset_long[dataset_long[id_col_name] == sid]
        cluster.reset_index(inplace=True)
        times_unique = list(np.unique(cluster["time"]))
        # Go through vars
        for varname in colnames:

            varname_stripped = varname.split("_t")[0]

            if varname_stripped not in time_dependent:
                # time-independent, so just use first value for this subject
                dataset_wide.loc[dfrow, varname_stripped] = cluster.at[0, varname]
            else:
                # time-dependent variable
                for t in times_unique:
                    dataset_wide.loc[dfrow, varname] = cluster.at[
                        np.where(cluster[time_col_name] == t)[0][0], varname_stripped
                    ]
    return dataset_wide


def make_long(dataset_wide):
    """
    Converts a wide dataset format to a long format.
    Arguments
    ---------
        dataset_wide: Dataframe
            input data
    Returns
    --------
    DataFrame
        dataframe in long data format
    """

    # Get long colnames
    #
    vars_and_times = [j.split("_t=") for j in dataset_wide.columns if "_t=" in j]
    vars_and_times = np.array(vars_and_times)
    # time independent first
    time_independent = [j for j in dataset_wide.columns if "_t=" not in j]
    colnames = time_independent.copy()
    # add time-dependent covars
    time_dependent = list(np.unique(vars_and_times[:, 0]))
    colnames.extend(time_dependent)
    # add time indicator
    colnames.append("time")

    # Initialize df
    dataset_long = df(columns=colnames)

    # Get unique time to look at
    times_unique = list(np.unique(vars_and_times[:, 1]))

    # Go through cluster and make long
    for clusteridx, cluster in dataset_wide.iterrows():
        print("Cluster {} of {}".format(clusteridx, dataset_wide.shape[0] - 1))
        for t in times_unique:

            row_to_add = df(columns=colnames)
            row_to_add.loc[0, "time"] = float(t)
            for varname in time_independent:
                row_to_add.loc[0, varname] = cluster[varname]
            for varname in time_dependent:
                try:
                    row_to_add.loc[0, varname] = cluster[varname + "_t=" + t]
                except KeyError:
                    # variable not recorded at this time point
                    continue

            dataset_long = concat((dataset_long, row_to_add), axis=0)

    return dataset_long


def code_categorical(single_variable):
    """Code categorical variable into numerical codes

    TODO: pd.get_dummies is probably better at this!!

    Arguments
    ---------
    single_variable: np.array
        numpy array of a single variables

    Returns
    -------
    np.array
        coded variable
    dict
     codes used
    """

    unique_values = set(single_variable)
    code_scheme = {}
    for i, j in enumerate(unique_values):
        if type(j) is str:

            if " " in j:
                j_fixed = j.replace(" ", "_")
            else:
                j_fixed = j
            code_scheme[i] = j_fixed
            single_variable[single_variable == j] = i

    single_variable = np.float32(single_variable)

    return single_variable, code_scheme


def make_dummy(single_variable, varname, code_scheme):
    """
    Convert a single, coded categorical variable into a dummy variables.

    Arguments
    ---------
    single_variable: np.array
        numpy array (coded categorical variable)
    varname: str
        name of variables
    code_scheme: dict
        a dict of names of codes
    Returns
    -------
    DataFrame
        dataframe with dummy variables.
    """
    dummyvars = df()
    cidx = 0
    for code in code_scheme:
        dummyvars[varname + "_" + str(code_scheme[code])] = 0 + (
            single_variable == code
        )

        cidx += 1
        # only use one dummy var for binary variables
        if cidx == 1 and len(code_scheme) == 2:
            break
    return dummyvars


def combine_vars(
    data_frame,
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
    data_frame: DataFrame
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
    new_var = data_frame.values[:, var_idxs]

    if operator_type == "OR":
        new_var = 0 + (np.sum(new_var, axis=1) > 0)
    elif operator_type == "AND":
        new_var = 0 + (np.sum(new_var, axis=1) == new_var.shape[1])
    else:
        raise ValueError("Unknown operator_type")

    if drop:
        for varname in var_names:
            data_frame = data_frame.drop(varname, axis=1)

    if newcolname is not None:
        data_frame[newcolname] = new_var
    else:
        data_frame[basestring] = new_var

    return data_frame
