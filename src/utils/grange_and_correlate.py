import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as stattools


def calculate_grangercausality(target, feature, maxlag=12):
    """Function to calcualte the granger causality for a feature and its target variable over the maxlags.

    Args:
        target (pandas.DataFraem): Data of the target variable. Index is Date.
        feature (pandas.DataFrame): Data of the feature. Index is Date.
        maxlag (int, optional): Number of lags to be applied. Defaults to 12.

    Returns:
        list, list: Returns two lists with the f-values and the p-values.
    """
    # Configuration for calculation
    lag_range = range(1, maxlag + 1)
    f_list = []
    p_list = []

    # Calculate causalities for all lags.
    for lag in lag_range:
        res = stattools.grangercausalitytests(
            pd.DataFrame(target.dropna()).join(feature.dropna(), how="inner"),
            maxlag=maxlag,
            verbose=False,
        )
        f, p, _, _ = res[lag][0]["ssr_ftest"]
        f_list.append(f)
        p_list.append(p)

    return f_list, p_list


def crosscorr(datax, datay, lag=0):
    """Lag-N cross correlation. Taken from: https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas

    Args:
        datax (pandas.Series): Data for the x values.
        datay (pandas.Series): Data for the y values.
        lag (int, optional): Number of lags to be applied. Defaults to 0.

    Returns:
        float: Croscorrelation between X and Y for n-lags.
    """
    return datax.corr(datay.shift(lag))


def grange_and_correlate(
    df, target, granger_lags=12, correlation_lags=12, number_of_features=10
):
    """Function to abstract the most important features for the target variable using a combination of granger causality and cross correlation.

    Args:
        df (pandas.DataFrame): Data to be evaluated.
        target (str): Name of the target feature column
        granger_lags (int, optional): Number of lags for the granger causality. Defaults to 12.
        correlation_lags (int, optional): Number of lags for the cross correlation. Defaults to 12.
        number_of_features (int, optional): Max number of features per method to be abstracted. If all unique should be 10. Defaults to 10.

    Returns:
        list, pandas.DataFrame, pandas.DataFrame: List of the relevant feature names, dataframe of the granger causalties and cross correlations.
    """
    # Filter the feature names list for the target
    feature_names = df.columns.tolist()
    feature_names = [name for name in feature_names if target not in name]

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   Calculate the causality:
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    df_granger = pd.DataFrame(columns=["fval", "pval", "feature_name"])
    for i in range(0, len(feature_names)):

        # Apply calculation function
        f_list, p_list = calculate_grangercausality(
            df[target], df[feature_names[i]], maxlag=granger_lags
        )

        # Generate dataset.
        df_eval = pd.DataFrame({"fval": f_list, "pval": p_list})

        df_eval_filtered = df_eval.sort_values("fval").reset_index(drop=True).iloc[:1]
        df_eval_filtered["feature_name"] = feature_names[i]

        df_granger = df_granger.append(df_eval_filtered)
    df_granger.reset_index(drop=True, inplace=True)
    # Abstract desired granger features.
    granger_features = (
        df_granger.sort_values("fval", ascending=False)
        .reset_index(drop=True)
        .loc[:number_of_features, "feature_name"]
        .tolist()
    )

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   Calculate the correlation:
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    df_corr = pd.DataFrame(columns=["lag_range", "abs_corr", "feature_name"])
    for i in range(0, len(feature_names)):

        # Apply cross correlation function
        xcov_monthly = [
            crosscorr(df[target], df[feature_names[i]], lag=lag)
            for lag in range(-correlation_lags, correlation_lags)
        ]
        xcov_monthly_abs = [abs(lag) for lag in xcov_monthly]
        range_list = [*range(-correlation_lags, correlation_lags)]

        # Combine the data.
        df_eval = pd.DataFrame({"lag_range": range_list, "abs_corr": xcov_monthly_abs})

        df_eval_filtered = (
            df_eval.sort_values("abs_corr").reset_index(drop=True).iloc[:1]
        )
        df_eval_filtered["feature_name"] = feature_names[i]

        df_corr = df_corr.append(df_eval_filtered)
    df_corr.reset_index(drop=True, inplace=True)
    # Abstract desired correlation features
    corr_features = (
        df_corr.sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
        .loc[:number_of_features, "feature_name"]
        .tolist()
    )

    # Combine the relevant features in list and get unique
    relevant_features = [
        item for sublist in zip(granger_features, corr_features) for item in sublist
    ]
    relevant_features = np.unique(relevant_features).tolist()

    return relevant_features, df_granger, df_corr


def user_input_crosscorr(df_target, df_user_input, max_lags=30):
    """Function that computes the cross correlation between target and user input features for max_lags.

    Args:
        df_target (pandas.Series): Input data series. Date is index.
        df_user_input (pandas.DataFrame): Target features from the user. Date is index.
        max_lags (int, optional): Number of lags to be used in the cross correlation. Defaults to 30.

    Returns:
        dict: Returns a dictionary with the features for names and the values as dataframes.
    """
    feature_names = df_user_input.columns.tolist()
    crosscorr_dict = {}
    for i in range(0, len(feature_names)):
        # Apply cross correlation function
        xcov_monthly = [
            crosscorr(df_target, df_user_input[feature_names[i]], lag=lag)
            for lag in range(-max_lags, max_lags)
        ]
        xcov_monthly_abs = [lag for lag in xcov_monthly]
        range_list = [*range(-max_lags, max_lags)]

        # Combine the data.
        df_eval = pd.DataFrame(
            {
                "lag_range": range_list,
                "correlation": xcov_monthly,
                "feature_name": [feature_names[i]] * len(range_list),
            }
        ).set_index("lag_range")

        crosscorr_dict[feature_names[i]] = df_eval

    return crosscorr_dict


def user_input_correlation_picker(
    df_target, df_user_input, selected_method="mean", max_lags=12
):
    """Function to compute the cross correlation for user input target variables returning the correlation values for all user input features.

    Args:
        df_target (pandas.Series): Target series to be correlated. Date is index.
        df_user_input (pandas.DataFrame): Feature frame to be correlated. Date is index.
        selected_lag (int, optional): Selected lag that is returned in the featuers.. Defaults to 0.
        max_lags (int, optional): Maximum lags that are used in the correlation. Selected lags must be within max_lags. Defaults to 30.

    Returns:
        dict: Returns the features dictionary to the target and their correlation value.
    """
    # TODO selected_lag to selected_method="mean"
    crosscorr_dict = user_input_crosscorr(
        df_target=df_target, df_user_input=df_user_input, max_lags=max_lags
    )
    feature_names = df_user_input.columns.tolist()

    if selected_method == "mean":
        correlation_dict = {}
        for feature in feature_names:
            correlation_dict[feature] = crosscorr_dict[feature]["correlation"].mean()
    elif selected_method == "max":
        correlation_dict = {}
        for feature in feature_names:
            correlation_dict[feature] = crosscorr_dict[feature]["correlation"].max()

    return correlation_dict
