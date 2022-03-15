# imports
from functools import partial
from warnings import filterwarnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import mean_squared_error

# LightGBM complains if you give it a pandas dataframe
filterwarnings(category=UserWarning, module="lightgbm", message=r".*peak memory cost.*", action="ignore")


# Feature engineering functions here


def lag_columns(df, columns, lags, fillna="bfill"):
    """Simple column lagger for pandas DataFrames with a DatetimeIndex index.

    Args:
        df (pandas.Dataframe): Dataframe, must have index of type DatetimeIndex
        columns (list): List of column names to lag.
        lags (list of int): List of lag intervals to apply.
        fillna (str or NoneType, optional): Fill method to use for missing values, used as the "method" argument to the DataFrame.fillna() method if not NoneType. Note that "bfill" introduces a temporal data leak, this may be acceptable if only a few missing values at the start of the lag columns are introduced, but watch out for columns which already have missing values elsewhere. Defaults to "bfill".

    Returns:
        pandas.DataFrame: The original column with the lagged columns added.
    """
    for col in columns:
        for lag in sorted(lags, reverse=True):
            # Create the new name of the column and drop it if it already exists.
            if lag > 0:
                newname = f"{col}_lag{lag}"
            elif lag < 0:
                newname = f"{col}_fstep{abs(lag)}"
            else:
                raise ValueError("Lag is neither positive nor negative?")
            df = df.drop(columns=newname, errors="ignore")
            # Create lagged column with the .shift() method, fill missing values, and insert it next to the original column.
            loc = df.columns.get_loc(col) + 1
            lagged_col = df[col].shift(lag)
            if fillna is not None:
                lagged_col = lagged_col.fillna(method=fillna)
            df.insert(loc=loc, column=newname, value=lagged_col)
    return df


def add_month_sin_cos(df, period_col, name_prefix, period=12):
    """Sin and cos encoding of periodic features with datetime dtype.

    Args:
        df (dataframe): dataframe to add features to.
        period_col (numpy.array): The values which the periodic encoding will be performed on. Note, these are the actual
        raw values, not just the name of the column. Use df.index.month.to_numpy() to get the month values from a datetimeindex.
        name_prefix (string): The prefix of the new columns, e.g. "month".
        period (int): The period length of the period_col values, e.g. 12 for months.

    Returns:
        dataframe: The original dataframe with the added columns.
    """
    df[f'{name_prefix}_sin'] = np.sin((period_col * 2 * np.pi) / period)
    df[f'{name_prefix}_cos'] = np.cos((period_col * 2 * np.pi) / period)
    return df


# Model selection-related functions


def rename_series_or_dataframe(s, new_name):
    if isinstance(s, pd.Series):
        s.name = new_name
    else:
        old_name = s.columns[-1]
        s = s.rename(columns={old_name: new_name})
    return s


def cross_val_forecast_dates_darts(model, series, dates_list, horizon, past_covariates=None, future_covariates=None,
                                   return_series=False):
    """
    Collect Darts model forecasts for a list of specific dates.

    Args:
        model (darts Model): Model used for forecasts, will be refit for each forecast.
        series (darts TimeSeries): The series which will be forecast.
        dates_list (list of pd.TimeSeries (string date may also work)): List of dates immediately preceding the forecasts.
        horizon (int): The forecast horizon, forecasts will be made for each point up to and including this many timepoints
        after each date.
        past_covariates (darts TimeSeries): past_covariates which will be passed to the model.
        future_covariates (darts TimeSeries): future_covariates which will be passed to the model.
        return_series (bool): Whether to return the forecasts as a TimeSeries rather than a pandas Series.

    Returns:
        Dict of dicts of results, indexed by the test dates.
    """
    # Cross-validation forecast function for time series
    d = {}

    # Make this somehow conform to cross_val_predict specifications
    for last_train_date in dates_list:
        d[last_train_date] = {}

        forecast_series = darts_date_predictions(series, model, last_train_date, horizon,
                                                 past_covariates=past_covariates, future_covariates=future_covariates)

        if return_series is True:
            d[last_train_date]["test"] = forecast_series
        else:
            y_pred = forecast_series.pd_series()
            y_pred = rename_series_or_dataframe(y_pred, "y_pred")
            d[last_train_date]["test"] = y_pred

    return d


def forecast_metrics_dict(forecasts: dict, y: pd.DataFrame):
    y = rename_series_or_dataframe(y, "y")
    def add_metrics(preds: pd.DataFrame, y: pd.DataFrame):
        if isinstance(preds, pd.DataFrame) is False:
            preds = pd.DataFrame(preds)
        preds = rename_series_or_dataframe(preds, "y_pred")
        preds = preds.join(y)
        preds['pred_err'] = preds['y_pred'] - preds['y']
        preds['abs_err'] = abs(preds['y_pred'] - preds['y'])
        preds['sq_err'] = (preds['y_pred'] - preds['y']) ** 2
        preds['perc_err'] = (abs(preds['y_pred'] - preds['y']) / preds['y'])
        preds['mape'] = mape(preds['y'], preds['y_pred'])
        return preds

    d1 = {}
    for key in forecasts:
        d2 = {}
        # forecast_start = train.index[-1].strftime("%Y-%m-%d")
        if "train" in forecasts[key]:
            d2["train"] = add_metrics(forecasts[key]["train"], y)
        d2["test"] = add_metrics(forecasts[key]["test"], y)
        d1[key] = d2

    return d1


def step_results_frame(metrics_dict: dict, column: str, table="test", print_results=False):
    store_list = []
    for start_date in metrics_dict:
        sd = {}
        sd["start"] = pd.to_datetime(start_date)
        for step, val in enumerate(metrics_dict[start_date][table][column], start=1):
            sd[f"{column}_s{step}"] = val
        store_list.append(sd)
    results = pd.DataFrame(store_list).set_index("start")
    if print_results == True:
        for m, v in results.mean(axis=0).iteritems():
            print(f"Mean {m} = {v}")
        print(f"Mean for all steps = {results.mean().mean()}")
    return results


def df_to_excel(df, fullpath, datetime_format="DD/MM/YYYY"):
    with pd.ExcelWriter(fullpath, datetime_format=datetime_format) as writer:
        df.to_excel(writer, index=True)
        print(f"Written to {fullpath}")


def results_xlsx_from_metrics_dict(metrics_dict, metric_list, filepath):
    with pd.ExcelWriter(filepath, datetime_format="DD/MM/YYYY") as writer:
        for column, display_name in metric_list:
            df = step_results_frame(metrics_dict, column, print_results=False)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime("%Y/%m/%d")
            df.to_excel(writer, sheet_name=display_name)
    print(f"File written to {filepath}")


def plot_step_rmse_bars(metrics_dict):
    results = step_results_frame(metrics_dict, "sq_err")
    rmses = np.sqrt(results.mean(0))
    ax = rmses.plot.bar(rot=0, figsize=(5, 8))
    _ = ax.set_xticklabels([s.replace("sq_err", "RMSE") for s in rmses.index])
    _ = ax.set_title(f"Mean RMSE = {np.mean(rmses):.2f}")
    sns.despine()
    return ax

def mape(y, y_pred):
    """Calculate the MAPE performance metric.

    Args:
        y (pandas.DataFrame): True "y" values.
        y_pred (pandas.DataFrame): Predicted "y" values.

    Returns:
        float64: Returns the calculated MAPE value.
    """
    return np.mean(np.abs((y - y_pred) / y)) * 100

def plot_backtest_forecasts(y_full, forecast_table_list, figsize=(14, 12), plot_before=24, shift=False):
    """
    Create a figure grid of backtest forecasts, where each plot shows the forecasts at a different horizon.
    Args:
        y_full (pandas Series): The original series which was forecasted.
        forecast_table (pandas Dataframe of shape n_timepoints x n_horizons):
        figsize (2-tuple): The figsize tuple which is passed to plt.subplots.
        plot_before (int): The length of y_full to plot before the backtest period.

    Returns:
        Matplotlib figure reference
    """
    forecast_table = pd.concat([f["forecast"] for f in forecast_table_list], axis=1)
    start_date = forecast_table.index[0]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    # rmse_func = partial(mean_squared_error, squared=False)
    for i in range(3):
        ax = axes[i]
        if shift == True:
            backtest_forecasts = forecast_table.iloc[:, i].shift(-i-1).dropna()
            start_date = forecast_table.index[0] # need to adjust start date after shifting
        else: 
            backtest_forecasts = forecast_table.iloc[:, i].dropna()
        xaxis_start = start_date - plot_before * start_date.freq
        xaxis_end = backtest_forecasts.index[-1]
        ax = sns.lineplot(data=y_full[xaxis_start:], ax=ax)
        ax = sns.lineplot(data=backtest_forecasts, ax=ax, marker=".")
        _ = ax.set_xlabel("Date")
        if not isinstance(y_full, pd.Series):
            y_full = y_full.iloc[:, 0]
        y_aligned, backtest_forecasts = y_full.align(backtest_forecasts, join="inner", axis=0)
        # rmse_score = rmse_func(y_full.reindex_like(backtest_forecasts).to_numpy(), backtest_forecasts.to_numpy())
        mape_val = mape(y_full.reindex_like(backtest_forecasts).to_numpy(), backtest_forecasts.to_numpy())
        _ = ax.set_title(f"Forecast horizon = {i + 1}, MAPE = {mape_val:.2f}")
        _ = ax.set_ylabel("")
        _ = plt.setp(ax.get_xticklabels(), ha="right", rotation=45)

    plt.tight_layout(h_pad=2)

    # props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    # _ = ax.text(x=0.02, y=0.5, s=f"{metric_text}{score:.2f}", transform=ax.transAxes,
    #             bbox=props)
    # _ = ax.set_title(f"Forecast after {start_date_text}")
    # _ = plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    sns.despine()
    return fig


def plot_forecast(metrics_dict, y_full, start_date, start_date_text, ax=None):
    frequency = pd.tseries.offsets.MonthBegin()
    rmse = partial(mean_squared_error, squared=False)
    metric_fun = rmse
    metric_text = "RMSE: "
    plot_before = 12  # Number of months to plot before the forecast period
    if ax is None:
        fig, ax = plt.subplots()
    y_test = metrics_dict[start_date]['test']['y']
    y_pred = metrics_dict[start_date]['test']['y_pred']
    y_pred = rename_series_or_dataframe(y_pred, "y forecast")
    score = metric_fun(y_test, y_pred)
    xaxis_start = max(pd.Timestamp(start_date) - plot_before * frequency, y_full.index.min())
    xaxis_end = y_test.index[-1]
    _ = y_full[xaxis_start:xaxis_end].plot(ax=ax)
    _ = y_pred.plot(marker=".", ax=ax)
    plt.legend()
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    _ = ax.text(x=0.02, y=0.5, s=f"{metric_text}{score:.2f}", transform=ax.transAxes,
                bbox=props)
    _ = ax.set_title(f"Forecast after {start_date_text}")
    _ = ax.set_xlabel(None)
    sns.despine()
    return ax


def plot_forecasts(metrics_dict, test_dates, y_full, fig_width, numcols, ax_height):
    # Plot forecasts for specific dates
    dates_to_plot = [d for d in metrics_dict if d in test_dates]
    numrows = int(np.ceil(len(dates_to_plot) / numcols))
    fig, axes = plt.subplots(numrows, numcols, figsize=(fig_width, ax_height * numrows), sharey=False)
    for i, start_date in enumerate(dates_to_plot):
        if axes.ndim == 2:
            axidx = i // numcols, i % numcols
        else:
            axidx = i
        ax = axes[axidx]
        start_date_text = start_date.strftime("%Y-%m")
        # fig, ax = plt.subplots(figsize=(8, 6))
        # res = gu.predictions_df_from_metrics_dict(metrics_dict, start_date)
        ax = plot_forecast(metrics_dict, y_full, start_date, start_date_text, ax=ax)
    for i in range(i + 1, numrows * numcols):
        axidx = i // numcols, i % numcols
        axes[axidx].set_axis_off()
    fig.tight_layout(h_pad=2)
    sns.despine()
    return fig


def darts_date_predictions(series, model, last_train_date, horizon, future_covariates=None, past_covariates=None):
    """
    Get forecasts after a specific date for a Darts model.
    """
    train, test = series.split_after(last_train_date)
    test_start_time = test.start_time()
    test = test.slice_n_points_after(test_start_time, horizon)
    fit_kwargs = {}
    predict_kwargs = {}
    if model.uses_future_covariates is True and future_covariates is not None:
        # OK to give models all future covariates, they will be internally aligned with the target
        fit_kwargs["future_covariates"] = future_covariates
        predict_kwargs["future_covariates"] = future_covariates
    if model.uses_past_covariates is True and past_covariates is not None:
        # Only give the training past covariates to the model, some Darts models will "cheat" and use future values if available
        fit_kwargs["past_covariates"] = past_covariates.split_after(last_train_date)[0]
    model.fit(train, **fit_kwargs)
    predictions = model.predict(horizon, **predict_kwargs)
    return predictions
