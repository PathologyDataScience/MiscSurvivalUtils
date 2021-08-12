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
