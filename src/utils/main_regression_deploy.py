import functools, pathlib, os, sys

import hyperopt
import hyperopt.hp as hp
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

utils_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(utils_location) not in sys.path:
    sys.path.append(os.path.realpath(utils_location))
import _feature_engineering as fe
import forecastutils as fu
import pipeline_model
import regressionforecasters as rf
from make_meta_features import create_meta_feature_frame

# Some utility functions
def forecaster_cross_val_score(
    regressor,
    X,
    target_series,
    timestep,
    cv,
    forecast_type,
    meta_feature_prefix=None,
    meta_features=None,
):
    """Returns a cross validation score for a SingleTimestepRegressionForecaster based on the scikit-learn pattern.

    Args:
        regressor (scikit-learn-like regressor): A regressor instance.
        X (pandas.DataFrame): Dataframe of y plus covariates.
        target_series (list of str): The name of the y column as a string
        timestep (int): The step size of the point to be forecast e.g. time t+timestep where t is the present
        cv (scikit-learn-like cross-validation object): e.g. TimeSeriesSplit
        forecast_type (bool): Sets whether the forecaster internally regresses the absolute value of the y or the forecast_type from the forecast from the present value.

    Returns:
        [int]: The RMSE of all forecasted values.
    """
    predictions_list = []
    # Shift indexes back so that all test y splits have equal size (forecaster returns future values).
    for train_idx, test_idx in cv.split(X):
        train_idx = (train_idx - timestep)[timestep:]
        test_idx = test_idx - timestep
        ssf = rf.SingleTimestepRegressionForecaster(
            regressor,
            timestep=timestep,
            forecast_type=forecast_type,
            meta_feature_prefix=meta_feature_prefix,
        )
        X_train = X.iloc[train_idx, :]
        X_test = X.iloc[test_idx, :]
        ssf = ssf.fit(X_train, target_series, meta_features=meta_features)
        y_pred = ssf.predict(X_test, meta_features=meta_features)
        predictions_list.append(y_pred)
    y_pred_all = pd.concat(predictions_list, axis=0)
    y_pred_all, y_aligned = y_pred_all.align(target_series, join="inner", axis=0)
    rmse_overall = sklearn.metrics.mean_squared_error(
        y_aligned, y_pred_all, squared=False
    )
    return rmse_overall


def test_hyperparams(
    regressor,
    timestep,
    X,
    target_series,
    cv,
    forecast_type,
    meta_feature_prefix=None,
    meta_features=None,
    hps=None,
):
    """Get the cross-validation score for a set of hyperparameters

    Args:
        regressor (scikit-learn-like regressor): Regressor object to be used as the final step of a pipeline.
        timestep (int): The step size of the point to be forecast e.g. time t+timestep where t is the present
        X (pandas.DataFrame): Dataframe of y plus covariates.
        target_series (list of str): The name of the y column as a string
        cv (scikit-learn-like cross-validation object): e.g. TimeSeriesSplit
        forecast_type (bool): Sets whether the forecaster internally regresses the absolute value of the y or the forecast_type from the forecast from the present value.
        hps (dict): The dictionary of hyperparameters to be passed to PipelineModel

    Returns:
        (int): The cross-validation RMSE result of the hyperparameters.
    """
    pm = pipeline_model.PipelineModel(
        hps=hps, estimator=regressor, multiout_wrapper=False
    )
    rmse = forecaster_cross_val_score(
        pm,
        X,
        target_series,
        timestep,
        cv,
        forecast_type,
        meta_feature_prefix,
        meta_features,
    )
    return rmse


def hyperopt_all_timesteps(
    regressor,
    horizon,
    X,
    target_series,
    space,
    cv,
    max_evals,
    forecast_type,
    meta_feature_prefix=None,
    meta_features=None,
):
    """Find the optimum hyperparameters for the forecasters for all timesteps using Hyperopt

    Args:


        regressor (scikit-learn-like regressor): Regressor object to be used as the final step of a pipeline.
        horizon (int): The range of timesteps into the future to be forecast (inclusive)
        X (pandas.DataFrame): Dataframe of y plus covariates.
        target_series (list of str): The name of the y column as a string
        space (dict): The dictionary specifying the hyperparameter space to be searched by Hyperopt.
        cv (scikit-learn-like cross-validation object): e.g. TimeSeriesSplit
        max_evals (int): The maximum number of evaluations to be performed by Hyperopt for each timestep.
        forecast_type (bool): Sets whether the forecaster internally regresses the absolute value of the y or the forecast_type from the forecast from the present value.

    Returns:
        [list of dict, (horizon, )]: A list of dicts containing the best hyperparameters for the pipeline at each timestep.
    """
    best_hyperparams_list = []
    for timestep in range(1, horizon + 1, 1):
        objective = functools.partial(
            test_hyperparams,
            regressor,
            timestep,
            X,
            target_series,
            cv,
            forecast_type,
            meta_feature_prefix,
            meta_features,
        )
        best = hyperopt.fmin(
            objective, space, algo=hyperopt.tpe.suggest, max_evals=max_evals
        )
        best_hyperparams_list.append(hyperopt.space_eval(space, best))
    return best_hyperparams_list


def wmape_error(y_true, y_pred):  # From the commodity desk utils folder
    wmape = np.sum(np.abs(y_pred.to_numpy() - y_true.to_numpy())) / np.sum(
        np.abs(y_true.to_numpy())
    )
    return wmape


def metrics_suite(y_true, y_pred, align=True):
    if align:
        y_true = y_true.dropna()
        y_pred = y_pred.dropna()
        y_true, y_pred = y_true.align(y_pred, join="inner", axis=0)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)
    wmape = wmape_error(y_true, y_pred)
    return mae, rmse, mape, wmape


def backtest_hyperparams(
    regressor,
    X,
    target_series,
    hyperparams_list,
    backtest_start_date=None,
    backtest_end_date=None,
    forecast_type=False,
    meta_feature_prefix=None,
    meta_features=None,
    deploy_mode=True,
):
    """
    Backtest the hyperparameters on the holdout period.
    """
    if deploy_mode:
        horizon = 1
    else:
        horizon = len(hyperparams_list)
    backtest_results_tables = []
    rmses = []
    mapes = []
    maes = []
    wmapes = []
    for i, timestep in tqdm(
        enumerate(range(1, horizon + 1, 1)),
        desc=f"Getting backtest results for timesteps",
    ):
        hyperparams = hyperparams_list[i]
        pm = pipeline_model.PipelineModel(
            hps=hyperparams, estimator=regressor, multiout_wrapper=False
        )
        ssf = rf.SingleTimestepRegressionForecaster(
            pm,
            timestep=timestep,
            forecast_type=forecast_type,
            meta_feature_prefix=meta_feature_prefix,
        )
        backtest_dates = X.index[
            (X.index >= backtest_start_date) & (X.index <= backtest_end_date)
        ]
        predictions_list = []
        for backtest_date in backtest_dates:
            X_train = X.loc[X.index < backtest_date, :]
            y_train = target_series.loc[target_series.index < backtest_date, :]
            ssf = ssf.fit(X_train, y_train, meta_features=meta_features)
            y_pred = ssf.predict(conf_ints=True)
            predictions_list.append(y_pred)
        y_backtest = pd.concat(predictions_list, axis=0)
        mae, rmse, mape, wmape = metrics_suite(target_series, y_backtest["forecast"])
        rmses.append(rmse)
        mapes.append(mape * 100)
        maes.append(mae)
        wmapes.append(wmape)
        backtest_results_tables.append(y_backtest)
    return backtest_results_tables, rmses, mapes, maes, wmapes


def get_train_results(
    regressor,
    X,
    target_series,
    hyperparams_list,
    forecast_type=False,
    meta_feature_prefix=None,
    meta_features=None,
    deploy_mode=True,
):
    """
    Train and test on the same. Only give the training set for X!
    """
    # horizon = len(hyperparams_list)
    if deploy_mode:
        horizon = 1
    else:
        horizon = len(hyperparams_list)
    train_results_tables = []
    rmses = []
    mapes = []
    maes = []
    wmapes = []
    for i, timestep in tqdm(
        enumerate(range(1, horizon + 1, 1)), desc="Getting results on training set"
    ):
        hyperparams = hyperparams_list[i]
        pm = pipeline_model.PipelineModel(
            hps=hyperparams, estimator=regressor, multiout_wrapper=False
        )
        ssf = rf.SingleTimestepRegressionForecaster(
            pm,
            timestep=timestep,
            forecast_type=forecast_type,
            meta_feature_prefix=meta_feature_prefix,
        )
        ssf = ssf.fit(X, target_series, meta_features=meta_features)
        forecast = ssf.predict(X, y=target_series, meta_features=meta_features)
        mae, rmse, mape, wmape = metrics_suite(target_series, forecast)
        rmses.append(rmse)
        mapes.append(mape * 100)
        maes.append(mae)
        wmapes.append(wmape)
        train_results_tables.append(forecast)
    return train_results_tables, rmses, mapes, maes, wmapes


def metrics_frame_single(timestep, set_name, maes, rmses, mapes, wmapes):
    values = [
        maes[timestep - 1],
        rmses[timestep - 1],
        mapes[timestep - 1],
        wmapes[timestep - 1],
    ]
    values_frame = pd.DataFrame(
        data=np.array(values).reshape((1, 4)),
        columns=["MAE", "RMSE", "MAPE", "WMAPE"],
        index=[set_name],
    )
    return values_frame


def main(df, target_name, horizon):
    df = df.dropna(subset=[target_name])
    target_series = df[[target_name]]

    # Set the variables below to enable the sequential meta-features regression
    meta_feature_prefix = "meta__ETS"
    meta_features = create_meta_feature_frame(
        df[target_name].dropna(), horizon, lead=24, ci_alpha=0.05
    )
    indicator_cols = [c for c in df.columns if c not in target_name]

    ########## Test parameters - should normally be left untouched #
    forecast_type = "proportional_diff"  # Whether the RegressionForecaster should internally regress absolute y values or differences.
    backtest_start_date = "2019-01"
    backtest_end_date = "2021-12"
    # Validation dates manually set to agree with the cross-validation splitter below
    validation_start_date = "2014-01"
    validation_end_date = "2018-12"
    # HYPEROPT_FMIN_SEED = 123
    # os.environ["HYPEROPT_FMIN_SEED"] = str(HYPEROPT_FMIN_SEED)
    #
    ####### Feature engineering below, adapt as desired ##############################
    ########################## multiplicative features #####################
    df = df.join(
        df[[target_name] + indicator_cols]
        .ewm(alpha=0.6)
        .mean()
        .pct_change()
        .add_suffix("_ewm_pct_change1")
    )
    df = df.join(
        df[[target_name] + indicator_cols]
        .ewm(alpha=0.6)
        .mean()
        .pct_change(periods=2)
        .add_suffix("_ewm_pct_change2")
    )
    ## setting: add skew
    skew_setting = [6]  # should be at least 3
    df, skewcols = fe.add_skew(
        df, features=[target_name] + indicator_cols, window=skew_setting
    )
    # Trigonometric encoding of months
    df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
    # Remove the raw indicators and target series
    df = df.drop(columns=[target_name] + indicator_cols)
    #
    ######### Hyperopt settings below ########
    #
    max_hyperopt_evals = 50
    cv = sklearn.model_selection.TimeSeriesSplit(20, test_size=3)  # Cross-validation
    #
    # Define the Hyperopt search space
    space = {
        # Feature Selector part
        "sel__estimator": {
            "type": "lgb.LGBMRegressor(boosting_type= 'gbdt',"
            " colsample_bytree=0.27, learning_rate=0.056,"
            " max_depth=7, n_estimators=200, num_leaves=16,"
            " subsample=0.37, subsample_freq= 5, random_state=42)"
        },
        "sel__n_features_to_select": hyperopt.pyll.scope.int(
            hp.quniform("sel__n_features_to_select", low=5, high=df.shape[1], q=1)
        ),
        "sel__step": 1,
        ## Final model part
        "est__n_estimators": 400,  # Make this lower if you're in a hurry, higher if you want better performance
        "est__colsample_bytree": hp.uniform("est__colsample_bytree", low=0.1, high=0.9),
        "est__num_leaves": hyperopt.pyll.scope.int(
            hp.quniform("est__num_leaves", low=10, high=50, q=1)
        ),
        "est__learning_rate": hp.loguniform(
            "est__learning_rate", low=-5 * np.log(10), high=0.5
        ),
        "est__subsample": hp.uniform("est__subsample", low=0.3, high=1.0),
        "est__subsample_freq": hyperopt.pyll.scope.int(
            hp.quniform("est__subsample_freq", low=1, high=10, q=1)
        ),
        # 'est__learning_rate': 0.01,
        "est__max_depth": hyperopt.pyll.scope.int(
            hp.quniform("est__max_depth", low=2, high=15, q=1)
        ),
        "est__random_state": 42,
        "est__boosting_type": "gbdt",
    }
    ############### All going well, you shouldn't have to adapt anything below this line ###############################
    #
    ############### Find hyperparameters with Hyperopt and backtest the hyperparameters on the holdout period ##########
    regressor = lgb.LGBMRegressor

    # Fit hyperparams on pre-backtest data
    X_train = df.loc[df.index < backtest_start_date]
    best_hyperparams_list = hyperopt_all_timesteps(
        regressor,
        horizon,
        X_train,
        target_series,
        space,
        cv,
        max_hyperopt_evals,
        forecast_type,
        meta_feature_prefix=meta_feature_prefix,
        meta_features=meta_features,
    )
    #
    ######### Return model performance
    # Train, validation and test metrics (4)
    # Get backtest and validation values using the backtest_hyperparams function, set dates appropriately
    # Get train metrics by training and predicting with the train set, one step

    (
        forecast_table_test,
        rmses_test,
        mapes_test,
        maes_test,
        wmapes_test,
    ) = backtest_hyperparams(
        regressor,
        df,
        target_series,
        best_hyperparams_list,
        backtest_start_date,
        backtest_end_date,
        forecast_type,
        meta_feature_prefix=meta_feature_prefix,
        meta_features=meta_features,
    )

    (
        forecast_table_val,
        rmses_val,
        mapes_val,
        maes_val,
        wmapes_val,
    ) = backtest_hyperparams(
        regressor,
        df,
        target_series,
        best_hyperparams_list,
        validation_start_date,
        validation_end_date,
        forecast_type,
        meta_feature_prefix=meta_feature_prefix,
        meta_features=meta_features,
    )

    # Get the training metrics by fitting and predicting on the pre-validation data
    (
        forecast_tables_train,
        rmses_train,
        mapes_train,
        maes_train,
        wmapes_train,
    ) = get_train_results(
        regressor,
        df.loc[df.index < validation_start_date],
        target_series,
        best_hyperparams_list,
        forecast_type=forecast_type,
        meta_feature_prefix=meta_feature_prefix,
        meta_features=meta_features,
    )

    values_frame_test = metrics_frame_single(
        1, "Test", rmses_test, mapes_test, maes_test, wmapes_test
    )
    values_frame_val = metrics_frame_single(
        1, "Validation", rmses_val, mapes_val, maes_val, wmapes_val
    )
    values_frame_train = metrics_frame_single(
        1, "Train", rmses_train, mapes_train, maes_train, wmapes_train
    )
    metrics_frame = pd.concat(
        (values_frame_train, values_frame_val, values_frame_test), axis=0
    )

    ############ Return forecast
    # Train a multitimestep model on all the data and create the forecast
    # Format: DT index (with all dates?), "forecast", "lower", "upper". Lower and upper optional.
    # Create models with the previously-optimized hyperparameters
    pipeline_list = [
        pipeline_model.PipelineModel(hps=h, estimator=regressor, multiout_wrapper=False)
        for h in best_hyperparams_list
    ]
    model = rf.MultipleTimestepRegressionForecaster(
        pipeline_list, horizon, forecast_type, meta_feature_prefix=meta_feature_prefix
    )
    model = model.fit(df, y=target_series, meta_features=meta_features)
    forecast = model.predict(conf_ints=True)
    forecast = pd.concat((target_series, forecast), axis=0).drop(columns=target_name)

    ############ Return feature importances
    ## Train a model on all the data and get the feature importances for the 1 month forecaster?
    # Format: columns "feature" with feature names and "importance" with the absolute values. Set "feature" as the index.
    feature_importances = model.forecasters[0].regressor.get_feature_importance(
        df.columns, [target_name]
    )
    feature_importances = feature_importances.rename(columns={"y": "importance"})
    feature_importances.index.name = "feature"

    ############ Return actuals
    # The target series, but index aligned with the forecast (same start date but end date does not extend past present)
    target_series.index.name = "Date"
    target_series = target_series.rename(columns={target_name: "actual"})
    # print(target_series)
    # print(forecast)
    # print(metrics_frame)
    backtesting = forecast_table_test[0]
    backtesting = target_series.join(backtesting, how="inner")

    return target_series, forecast, feature_importances, backtesting
