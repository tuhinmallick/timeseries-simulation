import logging
import os
import pickle
import shutil

import mlflow
import seaborn as sns
from matplotlib import pyplot as plt

# Import custom libraries

# # Use versions of libraries from commodity desk now
# if r"C:\Users\SutcliG\source\commodity-desk-modeling\models\SmoothBoost" not in sys.path:
#     sys.path.insert(0, r"C:\Users\SutcliG\source\commodity-desk-modeling\models\SmoothBoost")

# if os.path.realpath("pipelib") not in sys.path:
#     sys.path.append(os.path.realpath("pipelib"))

pipelib_path = r"C:\Users\SutcliG\source\commodity-desk-modeling\models\SmoothBoost\pipelib"

from BASF_Metals.Models.src.smoothboost.main_regression_deploy import *

# Plot aesthetics
sns.set_style("dark")
plt.style.use("dark_background")
plt.rcParams['figure.facecolor'] = '#151934'
plt.rcParams['axes.facecolor'] = '#151934'
sns.set_palette('pastel')
sns.set_context('talk')

# Some utility functions

def forecast_dates(model, X, target_series, dates_list, meta_features=None):
    # Get forecasts for a list of starting point dates using a regression forecaster passed as an argument.
    d = {}
    for last_train_date in dates_list:
        d[last_train_date] = {}
        X_train = X.loc[X.index <= last_train_date]
        y_train = target_series.loc[target_series.index <= last_train_date]
        model.fit(X_train, y_train, meta_features=meta_features)
        forecast = model.predict()
        forecast = fu.rename_series_or_dataframe(forecast, "y_pred")
        d[last_train_date]["test"] = forecast
    return d

def add_residuals(forecast_table, target_series):
    resdf = pd.concat((forecast_table["forecast"], target_series), join="inner", axis=1)
    resdf = resdf.dropna()
    resdf["resid"] = resdf["forecast"] - resdf[resdf.columns[1]]
    return resdf

def save_residual_frames(forecast_tables_train, forecast_tables_val, forecast_tables_test):
    _ = os.makedirs("outputs/residuals")
    for timestep in range(1,horizon+1, 1):
        resid_train = add_residuals(forecast_tables_train[timestep-1].iloc[:-1,:], target_series)
        resid_train.to_csv(f"outputs/residuals/residuals_train_t{timestep}.csv")
        resid_val = add_residuals(forecast_tables_val[timestep-1], target_series)
        resid_val.to_csv(f"outputs/residuals/residuals_val_t{timestep}.csv")
        resid_test = add_residuals(forecast_tables_test[timestep-1], target_series)
        resid_test.to_csv(f"outputs/residuals/residuals_test_t{timestep}.csv")

if __name__ == "__main__":
    #
    # Create fresh outputs directory for artifacts to be logged by MLFlow
    if os.path.exists("outputs"): shutil.rmtree("outputs")  # Delete any previous artifacts
    os.makedirs("outputs")
    os.makedirs("outputs/backtest")
    os.makedirs("outputs/val_dates")
    os.makedirs("outputs/val_dates/forecast_plots")
    #
    #
    #
    ########## Test parameters - should normally be left untouched #
    horizon = 3
    forecast_type = "proportional_diff"  # Whether the RegressionForecaster should internally regress absolute y values or differences.
    backtest_start_date = "2019-01"
    backtest_end_date = "2021-12"
    HYPEROPT_FMIN_SEED = 123
    os.environ["HYPEROPT_FMIN_SEED"] = str(HYPEROPT_FMIN_SEED)
    # Validation dates manually set to agree with the cross-validation splitter below
    validation_start_date = "2014-01"
    validation_end_date = "2018-12"
    #
    #
    #
    ########## Change area below to load a different dataset ############
    #
    #
    #
    # Set mlflow IDs
    mlflow.end_run()  # In case the last run was not ended
    mlflow.set_experiment("Steel PPI")
    mlflow.start_run()  # Timing of run starts from here
    mlflow.set_tag("Target", "Steel PPI")
    # Load data
    target_name = "WPU101704"
    df = pd.read_csv(r"../data/input/steel_w_indicators_expanded.csv", index_col=0, parse_dates=True,
                     encoding='latin-1')
    df.index.freq = "MS"
    df = df[[target_name] + ["PCOALAUUSDM", "A31ANO", "A31ATI", "WPU1012", "VIX_monthly"]]
    df = df.dropna(subset=[target_name])
    target_series = df[[target_name]]


    # meta_feature_prefix=None
    # meta_features=None
    # Set the variables below to enable the sequential meta-features regression
    # meta_features_path = "meta_features_ETS.csv"
    # meta_features = pd.read_csv(meta_features_path, index_col=0, parse_dates=True)
    meta_features = create_meta_feature_frame(df[target_name].dropna(), horizon, ci_alpha=0.05)
    meta_feature_prefix = "meta__ETS"
    #
    #
    ######### Leave this ###############################################################################################
    # # Rename y column to "y" for universality
    # df = df.rename(columns={target_name[0]: "y"})
    # target_name = ["y"]
    # target_series = df[[target_name]].dropna()

    indicator_cols = [c for c in df.columns if c not in target_series]
    ####################################################################################################################
    #
    #
    #
    ####### Feature engineering below, adapt as desired ##############################
    #
    #
    #
    # # Feature engineering
    # df = gu.lag_columns(df, y, lags=list(range(1, 7)))  # 6 lags for the y
    # df = gu.lag_columns(df, indicator_cols, lags=[1, 2, 3])

    # ################## additive features ###################################################
    # These are appropriate if the forecast_type parameters is set to "additive_diff" or "absolute"
    #
    # # variable with original column names
    # column_names = df.columns
    #
    # time_columns = ["month_sin", "month_cos"]
    # ## inplacely add lags
    # lag_setting = [1, 2, 3, 6]
    # lagdf = fe.create_lag(df, features=column_names, lag=lag_setting, droplagna=False)
    # df = pd.concat([df, lagdf], axis=1).dropna()
    #
    # ## add ma / ema
    # ma_setting = [3, 6]
    # df, macols = fe.add_MA(df, features=column_names.append(lagdf.columns), window=ma_setting)
    # ema_setting = [.2, .4, .6, .8]
    #
    # ## setting: add std
    # ms_setting = [3, 6]
    # df, stdcols = fe.add_MS(df, features=column_names.append(lagdf.columns), window=ms_setting)
    #
    # ## setting: add skew
    # skew_setting = [3, 6]  # should be at least 3
    # df, skewcols = fe.add_skew(df, features=column_names.append(lagdf.columns), window=skew_setting)
    #
    # ## setting:  also get decomposition on exog features (STL model will auto decompose the endog features)
    # rolling = 24
    # stlhelper = STLModel.STLModel(horizon=horizon)
    # df = stlhelper._make_STL(
    #     df, target=column_names,
    #     freq="MS", rolling_strategy='rolling', rolling=rolling, extract_strategy='nanmean'
    # )
    # # Trigonometric encoding of months
    # df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
    #
    ########################### end additive features #######################

    ########################## multiplicative features #####################
    # These are appropriate if forecast_type is set to "proportional_diff"
    df = df.join(df[[target_name] + indicator_cols].ewm(alpha=0.6).mean().pct_change().add_suffix("_ewm_pct_change1"))
    df = df.join(
        df[[target_name] + indicator_cols].ewm(alpha=0.6).mean().pct_change(periods=2).add_suffix("_ewm_pct_change2"))
    # df = df.join(df[[target_name] + indicator_cols].ewm(alpha=0.6).mean().pct_change(periods=3).add_suffix("_ewm_pct_change3"))
    ## setting: add skew
    skew_setting = [6]  # should be at least 3
    df, skewcols = fe.add_skew(df, features=[target_name] + indicator_cols, window=skew_setting)
    # Trigonometric encoding of months
    df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
    # Drop the original raw indicators and target series to improve performance
    df = df.drop(columns=[target_name] + indicator_cols)
    #
    #
    ######### Hyperopt settings below ########
    #
    #
    #
    max_hyperopt_evals = 50
    cv = sklearn.model_selection.TimeSeriesSplit(20, test_size=3)  # Cross-validation
    #
    # Fit hyperparameters of LightGBM regression forecasters
    # Define the Hyperopt search space
    #
    space = {
        # Feature Selector part
        'sel__estimator': {
            "type": "lgb.LGBMRegressor(boosting_type= 'gbdt'," \
                    " colsample_bytree=0.27, learning_rate=0.056," \
                    " max_depth=7, n_estimators=200, num_leaves=16,"
                    " subsample=0.37, subsample_freq= 5, random_state=42)"}
        , 'sel__n_features_to_select': hyperopt.pyll.scope.int(
            hp.quniform('sel__n_features_to_select', low=5, high=df.shape[1], q=1)
        ),
        # 'sel__step': hp.choice('sel__step', [1]),
        'sel__step': 1,
        ## Final model part
        # , 'est__n_estimators': hyperopt.pyll.scope.int(
        #     hp.quniform('est__n_estimators', low=100, high=500, q=25)
        # ),
        'est__n_estimators': 400,
        'est__colsample_bytree': hp.uniform('est__colsample_bytree', low=0.1, high=0.9),
        'est__num_leaves': hyperopt.pyll.scope.int(hp.quniform('est__num_leaves', low=10, high=50, q=1)),
        # Setting learning_rate and n_estimators is mainly a matter of how long you're willing to wait,
        # keep learning_rate high & n_estimators low during development for speed then increase for final results
        'est__learning_rate': hp.loguniform('est__learning_rate', low=-5 * np.log(10), high=0.5),
        'est__subsample': hp.uniform('est__subsample', low=0.3, high=1.0),
        'est__subsample_freq': hyperopt.pyll.scope.int(hp.quniform('est__subsample_freq', low=1, high=10, q=1)),
        # 'est__learning_rate': 0.01,
        'est__max_depth': hyperopt.pyll.scope.int(hp.quniform('est__max_depth', low=2, high=15, q=1)),
        'est__random_state': 42,
        'est__boosting_type': 'gbdt'
    }
    #
    #
    #
    ############### All going well, you shouldn't have to adapt anything below this line ###############################
    #
    #
    #
    ############### Find hyperparameters with Hyperopt and backtest the hyperparameters on the holdout period ##########
    regressor = lgb.LGBMRegressor

    # Fit hyperparams on pre-backtest data
    X_train = df.loc[df.index < backtest_start_date]
    best_hyperparams_list = hyperopt_all_timesteps(regressor, horizon, X_train, target_series, space, cv,
                                                   max_hyperopt_evals,
                                                   forecast_type, meta_feature_prefix=meta_feature_prefix,
                                                   meta_features=meta_features)

    # Can disable hyperparameter search and use pre-calculated hyperparameters to save time during testing
    # best_hyperparams_list = [{'est__boosting_type': 'gbdt',
    #   'est__colsample_bytree': 0.3172304692605816,
    #   'est__learning_rate': 0.011682794123694116,
    #   'est__max_depth': 3,
    #   'est__n_estimators': 400,
    #   'est__num_leaves': 13,
    #   'est__random_state': 42,
    #   'est__subsample': 0.4477283603331486,
    #   'est__subsample_freq': 5,
    #   'sel__estimator': {
    #       'type': "lgb.LGBMRegressor(boosting_type= 'gbdt', colsample_bytree=0.4, learning_rate=0.0007, max_depth=15, n_estimators=200, num_leaves=39, random_state=42)"},
    #   'sel__n_features_to_select': 13,
    #   'sel__step': 1},
    #  {'est__boosting_type': 'gbdt',
    #   'est__colsample_bytree': 0.6110272570323517,
    #   'est__learning_rate': 0.01128260694968849,
    #   'est__max_depth': 7,
    #   'est__n_estimators': 400,
    #   'est__num_leaves': 12,
    #   'est__random_state': 42,
    #   'est__subsample': 0.3041644970242154,
    #   'est__subsample_freq': 1,
    #   'sel__estimator': {
    #       'type': "lgb.LGBMRegressor(boosting_type= 'gbdt', colsample_bytree=0.4, learning_rate=0.0007, max_depth=15, n_estimators=200, num_leaves=39, random_state=42)"},
    #   'sel__n_features_to_select': 9,
    #   'sel__step': 1},
    #  {'est__boosting_type': 'gbdt',
    #   'est__colsample_bytree': 0.8441041876514987,
    #   'est__learning_rate': 0.008900642976703876,
    #   'est__max_depth': 7,
    #   'est__n_estimators': 400,
    #   'est__num_leaves': 10,
    #   'est__random_state': 42,
    #   'est__subsample': 0.37282019783813375,
    #   'est__subsample_freq': 7,
    #   'sel__estimator': {
    #       'type': "lgb.LGBMRegressor(boosting_type= 'gbdt', colsample_bytree=0.4, learning_rate=0.0007, max_depth=15, n_estimators=200, num_leaves=39, random_state=42)"},
    #   'sel__n_features_to_select': 14,
    #   'sel__step': 1}]

    # Get also train and validation results, for residual calculation
    forecast_tables_val, rmses_val, mapes_val, maes_val, wmapes_val = backtest_hyperparams(
        regressor, df, target_series,
        best_hyperparams_list,
        validation_start_date,
        validation_end_date, forecast_type,
        meta_feature_prefix=meta_feature_prefix,
        meta_features=meta_features,
        deploy_mode=False
    )

    forecast_tables_train, rmses_train, mapes_train, maes_train, wmapes_train = get_train_results(
        regressor, df.loc[df.index < validation_start_date], target_series, best_hyperparams_list,
        forecast_type=forecast_type, meta_feature_prefix=meta_feature_prefix, meta_features=meta_features,
        deploy_mode=False
    )

    # Get backtest results THIS HAS TO BE AFTER THE TRAIN AND VAL RESULTS SO THE CORRECT RMSES & MAPES ARE LOGGED
    forecast_tables_test, rmses, mapes, maes, wmapes = backtest_hyperparams(regressor, df, target_series,
                                                                       best_hyperparams_list,
                                                                       backtest_start_date,
                                                                       backtest_end_date, forecast_type,
                                                                       meta_feature_prefix=meta_feature_prefix,
                                                                       meta_features=meta_features,
                                                                       deploy_mode=False)

    ################# Log parameters, metrics and backtest results to MLflow ##########################################
    save_residual_frames(forecast_tables_train, forecast_tables_val, forecast_tables_test)

    #
    mlflow.log_param("Backtest start", backtest_start_date)
    mlflow.log_param("horizon", horizon)
    mlflow.log_param("forecast_type", forecast_type)
    mlflow.log_param("max_hyperopt_evals", max_hyperopt_evals)
    mlflow.set_tag("Model type", "LightGBM regression forecaster")
    #
    # Log parameters
    for i, d in enumerate(best_hyperparams_list, start=1):
        mlflow.log_dict(d, f"hyperparams_timestep{i}.json")

    # Log metrics
    mlflow.log_metrics({f"MAPE_s{i + 1}": v for i, v in enumerate(mapes)})
    mlflow.log_metric("Mean_MAPE_all_steps", np.mean(mapes))
    mlflow.log_metrics({f"RMSE_s{i + 1}": v for i, v in enumerate(rmses)})
    mlflow.log_metric("Mean_RMSE_all_steps", np.mean(rmses))
    # Log input features and timespan of input
    mlflow.log_dict({"columns": df.columns.to_list()}, "columns.json")
    mlflow.log_dict({"index": [i.strftime("%Y-%m") for i in df.index]}, "index.json")

    # Create and save figures
    # Barplot of backtest results
    rmse_series = pd.Series(rmses, index=["H1", "H2", "H3"])
    fig, ax = plt.subplots(figsize=(5, 8))
    ax = rmse_series.plot.bar(rot=0, ax=ax)
    _ = ax.set_title(f"Overall RMSE = {np.mean(rmses):.2f}")
    _ = ax.set_ylabel("RMSE")
    plt.tight_layout()
    sns.despine()
    # Save figure to outputs directory as mlflow.log_figure() overrides background colour
    fig.savefig("outputs/backtest/barplot_rmse_backtest.svg", facecolor=fig.get_facecolor(), edgecolor="none")

    # Create and save tables of forecasted values
    for i, table in enumerate(forecast_tables_test):
        table.to_csv(f"outputs/backtest/backtest_output_table_t{i+1}.csv")

    # Triple barplot of backtest forecasts by horizon
    fig = fu.plot_backtest_forecasts(target_series, forecast_tables_test, figsize=(8, 8))
    fig.savefig("outputs/backtest/backtest_forecasts_plot.svg", facecolor=fig.get_facecolor(), edgecolor="none")
    #
    #
    #
    ############ Get 3-month backtest forecasts from Ferrovial's validation dates and save output ######################
    # Specify dates
    test_dates = ["2007-05", "2007-06", "2007-07", "2007-10", "2008-09", "2008-12", "2009-12", "2011-03", "2011-04",
                  "2014-04", "2014-09", "2015-07", "2016-04", "2016-05", "2018-01", "2018-02", "2018-08", "2019-03",
                  "2020-03", "2020-09", "2020-10", "2020-11", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05",
                  "2021-06", "2021-07", "2021-08", "2021-09", "2021-10", "2021-11"]
    # Create models with the previously-optimized hyperparameters
    pipeline_list = [pipeline_model.PipelineModel(hps=h, estimator=regressor, multiout_wrapper=False)
                     for h in best_hyperparams_list]
    model = rf.MultipleTimestepRegressionForecaster(pipeline_list, horizon, forecast_type,
                                                    meta_feature_prefix=meta_feature_prefix)
    forecasts_dict = forecast_dates(model, df, target_series, test_dates, meta_features=meta_features)
    # Save the forecasts to an excel spreadsheet
    metrics_dict = fu.forecast_metrics_dict(forecasts_dict, target_series)
    fu.results_xlsx_from_metrics_dict(metrics_dict, [("y_pred", "predicted value"), ("pred_err", "prediction error"),
                                                     ("abs_err", "absolute error")],
                                      "outputs/val_dates/forecast_data.xlsx")
    # Make a bar plot of mean absolute error for the validation dates
    results_dates = fu.step_results_frame(metrics_dict, "abs_err")
    fig, ax = plt.subplots(figsize=(13, 7))
    results_dates = results_dates.loc[~results_dates.isna().any(axis=1), :]
    ax = results_dates.mean(axis=1).plot.bar(ax=ax)
    _ = ax.set_title(f"Mean absolute error by forecast start date")
    _ = ax.set_xlabel("Date")
    _ = ax.set_xticklabels(results_dates.index.strftime("%Y-%m"), rotation=45)
    _ = fig.tight_layout()
    sns.despine()
    fig.savefig("outputs/val_dates/barplot_mae_dates.svg", facecolor=fig.get_facecolor(), edgecolor="none")
    # Bar plot of RMSEs for the validation dates
    results_dates = fu.step_results_frame(metrics_dict, "sq_err")
    rmses = np.sqrt(results_dates.mean(0))
    fig, ax = plt.subplots(figsize=(5, 8))
    ax = rmses.plot.bar(rot=0, ax=ax)
    _ = ax.set_xticklabels([s.replace("sq_err", "RMSE") for s in rmses.index])
    _ = ax.set_title(f"Mean RMSE = {np.mean(rmses):.2f}")
    _ = fig.tight_layout()
    sns.despine()
    fig.savefig("outputs/val_dates/barplot_rmse_dates.svg", facecolor=fig.get_facecolor(), edgecolor="none")
    # Log validation date rmses
    mlflow.log_metrics({f"RMSE_val_dates_s{i + 1}": v for i, v in enumerate(rmses)})
    mlflow.log_metric("Mean_val_date_RMSE_all_steps", np.mean(rmses))
    # Save a figure showing each of the validation forecasts
    for d in test_dates:
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            _ = fu.plot_forecast(metrics_dict, target_series, d, d, ax=ax)
            _ = fig.savefig(f"outputs/val_dates/forecast_plots/forecast_{d}.svg", facecolor=fig.get_facecolor(),
                            edgecolor="none")
            plt.close()
        except:
            logging.debug(f"Exception encountered when trying to plot forecast for {d}")
    #
    # Plot a final figure showing the future forecast with confidence intervals
    pipeline_list = [pipeline_model.PipelineModel(hps=h, estimator=regressor, multiout_wrapper=False)
                     for h in best_hyperparams_list]
    model = rf.MultipleTimestepRegressionForecaster(
        pipeline_list, horizon, forecast_type,
        meta_feature_prefix=meta_feature_prefix
    )
    model = model.fit(df, y=target_series, meta_features=meta_features)
    forecast = model.predict(conf_ints=True)
    forecast = pd.concat((target_series, forecast), axis=0)
    forecast.loc[target_series.index.max(), :] = target_series.loc[target_series.index.max()][0]
    fig, ax = plt.subplots(figsize=(12, 8))
    forecast = forecast["2021":]
    _ = forecast[target_name].plot(ax=ax)
    _ = forecast["forecast"].plot(ax=ax)
    _ = ax.fill_between(forecast.index, forecast["lower"], forecast["upper"], alpha=0.5, color='tab:orange')
    _ = fig.savefig("outputs/final_forecast.svg", facecolor=fig.get_facecolor(), edgecolor=None)
    ########### Tests complete, log model and other output with MLFLow #################################################
    # model = model.fit(df, y=target_series, meta_features=meta_features)
    if "meta_features" in locals():
        meta_features.to_csv("outputs/meta_features.csv")
    for i in range(horizon):
        feature_importances = model.forecasters[i].regressor.get_feature_importance(df.columns, [target_name])
        feature_importances.to_csv(f"outputs/feature_importances_t{i + 1}.csv")
    pickle.dump(model, open("outputs/model.pkl", "wb"))

    # Save scripts
    shutil.rmtree(os.path.join(pipelib_path, "__pycache__"), ignore_errors=True)  # Delete this to save space
    shutil.rmtree(os.path.join(pipelib_path, "kaisutils/__pycache__"), ignore_errors=True)  # Delete this to save space
    shutil.copy(os.path.realpath(__file__), "outputs")
    shutil.copytree(pipelib_path, "outputs/pipelib")

    # Save the conda environment
    os.system("conda list > outputs/env.txt")

    # Log all artifacts in the outputs directory then delete it
    mlflow.log_artifacts("outputs")
    shutil.rmtree("outputs")
    mlflow.end_run()
    print("It's over!")