# =======================================================================================
#   IMPORT LIBRARIES
# =======================================================================================

import os
from pickletools import TAKEN_FROM_ARGUMENT1
import sys
import logging
import pickle
# import _pickle as cPickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core
from azureml.core import Workspace

import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from minio.error import S3Error
import cloudpickle


base_path = r"C:\Users\mallict\forecasty-lab"

# base_path = r"C:\Users\OchsP\Documents\Python-dev\forecasty-lab"

if os.path.realpath(base_path) not in sys.path:
    sys.path.append(os.path.realpath(base_path))
    
from BASF_Metals.Models.src.smoothboost.main_regression_deploy import *
from BASF_Metals.Models.src.smoothboost.pipelib.reduce_memory_usage import reduce_mem_usage
from BASF_Metals.Models.src.smoothboost.pipelib.grange_and_correlate import grange_and_correlate, user_input_correlation_picker
from BASF_Metals.Models.src.smoothboost.pipelib.remove_missing_data import remove_features_with_na
from BASF_Metals.Models.src.smoothboost.pipelib.space import space_smoothboost
from BASF_Metals.Models.src.smoothboost.pipelib.mlflow_tracker import mlflow_settings as mf




# =======================================================================================
#   SUPPORTING FUNCTIONS
# =======================================================================================

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
    _ = os.makedirs("BASF_Metals/Models/notebooks/outputs/residuals")
    for timestep in range(1,horizon+1, 1):
        resid_train = add_residuals(forecast_tables_train[timestep-1].iloc[:-1,:], target_series)
        resid_train.to_csv(f"BASF_Metals/Models/notebooks/outputs/residuals/residuals_train_t{timestep}.csv")
        resid_val = add_residuals(forecast_tables_val[timestep-1], target_series)
        resid_val.to_csv(f"BASF_Metals/Models/notebooks/outputs/residuals/residuals_val_t{timestep}.csv")
        resid_test = add_residuals(forecast_tables_test[timestep-1], target_series)
        resid_test.to_csv(f"BASF_Metals/Models/notebooks/outputs/residuals/residuals_test_t{timestep}.csv")

def _mlflow(target_name):
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   CONFIGURE MLFLOW
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    import pdb; pdb.set_trace()
    subscription_id = 'fc563dde-a9ad-4edf-9998-2d52ba8afff9'
    # Azure Machine Learning resource group 
    resource_group = 'forecasty_ml_test' 

    # Azure Machine Learning workspace name
    workspace_name = 'forecasty_ml_workplace'

    # Instantiate Azure Machine Learning workspace
    ws = Workspace.get(name=workspace_name,
                    subscription_id=subscription_id,
                    resource_group=resource_group)
    #Set MLflow experiment. 
    experiment_name = f"{target_name} Forecast"
    # mf.set_env_vars()       # Setting the environment variables
    # mf.create_s3_bucket()   # Create the S3 bucket in MinIO if it doesn't exist
    # registry_uri =  r'sqlite:///C:\Users\mallict\forecasty-lab\BASF_Metals\artifacts\mlflow_db.db' #'sqlite:///mlflow_db.db'
    tracking_uri = ws.get_mlflow_tracking_uri()
    print(tracking_uri)
    # mlflow.tracking.set_registry_uri(registry_uri)
    mlflow.tracking.set_tracking_uri(tracking_uri)
    try:
            if not mlflow.get_experiment_by_name(experiment_name):
                mlflow.create_experiment(experiment_name, artifact_location='s3://mlflow')
    except MlflowException as err:
            print(err)    
    mlflow.set_experiment(experiment_name)      #Set experiment
    mlflow.set_tag("Target", f"{target_name} Spot Price")
    mlflow.end_run()  # In case the last run was not ended



# =======================================================================================
#   SET PLOT AESTHETICS
# =======================================================================================

sns.set_style("dark")
plt.style.use("dark_background")
plt.rcParams['figure.facecolor'] = '#151934'
plt.rcParams['axes.facecolor'] = '#151934'
plt.rcParams.update({'axes.facecolor':'#151934'})
sns.set_palette('pastel')
sns.set_context('talk')

if __name__ == "__main__":
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   KEY PARAMETERS
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    target_name = "Platinum"
    target = f"{target_name}_spot_price" # This is used as column selector.
    horizon = 6
    
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   CREATE TARGET FOLDERS
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    if os.path.exists("BASF_Metals/Models/notebooks/outputs"): shutil.rmtree("BASF_Metals/Models/notebooks/outputs")  # Delete any previous artifacts
    os.makedirs("BASF_Metals/Models/notebooks/outputs")
    os.makedirs("BASF_Metals/Models/notebooks/outputs/backtest")
    os.makedirs("BASF_Metals/Models/notebooks/outputs/val_dates")
    os.makedirs("BASF_Metals/Models/notebooks/outputs/val_dates/forecast_plots")
    os.makedirs("BASF_Metals/Models/notebooks/outputs/results_run")
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #   CONFIGURE MLFLOW
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    _mlflow(target_name)
    with mlflow.start_run(run_name='single_timestep_run') as single_timestep_run :  # Timing of run starts from here
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   SET CONFIGURATIONS FOR MODEL
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # Smooth Boost Config

        forecast_type = "proportional_diff" # Whether the RegressionForecaster should internally regress absolute y values or differences.

        backtest_start_date = "2020-01"
        backtest_end_date = "2021-12"

        HYPEROPT_FMIN_SEED = 123
        os.environ["HYPEROPT_FMIN_SEED"] = str(HYPEROPT_FMIN_SEED)

        # Validation dates manually set to agree with the cross-validation splitter below
        validation_start_date = "2014-01"
        validation_end_date = "2018-12"
        
        # Set the exfcel parameters.
        file_name = "Platinum_Drivers_Month"
        sheet_names = ["Data Dictionary", "Compiled Dataset"]
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   READ AND CONFIGURE THE DATA
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        # =========================== RAW UNTRANSFORMED DATA ===========================
        # Get the base data and adjust.
        df = pd.read_excel(
            os.path.join(base_path, "BASF_Metals", "raw_data", file_name + ".xlsx"),
            sheet_names[1]
            )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)

        # Read the data dictionary with the full names.
        df_dict = pd.read_excel(
            os.path.join(base_path, "BASF_Metals", "raw_data", file_name + ".xlsx"),
            sheet_names[0]
            )[["Feature Name", "Full Name"]]
        
        # Rename the target
        # df.columns = df_dict["Full Name"].values.tolist()
        
        # =========================== TIDY DATA ===========================
        # Remove features that contain a lot of missing data.
        df = remove_features_with_na(df, threshold=65, trailing_value=5)
        df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
        import pdb; pdb.set_trace()

        # Get preselected features and other causality and correlation metrics
        pre_selected_features, df_granger, df_correlation = grange_and_correlate(df, target, granger_lags=12, correlation_lags=12, number_of_features=10)
        
        
        df = df.dropna(subset=[target])
        target_series = df[[target]]
        
        # =========================== SIMULATION PREP ===========================
        simulation_correlation_dict = user_input_correlation_picker(df[target], df.drop([target], axis=1), selected_method="mean", max_lags=30)
        
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   SET META FEATURES
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        # meta_feature_prefix=None
        # meta_features=None
        # Set the variables below to enable the sequential meta-features regression
        # meta_features_path = "meta_features_ETS.csv"
        # meta_features = pd.read_csv(meta_features_path, index_col=0, parse_dates=True)
        meta_features = create_meta_feature_frame(df[target].dropna(), horizon, ci_alpha=0.05)
        meta_feature_prefix = "meta__ETS"

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   FEATURE ENGINEERING
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        ######### Leave this ###############################################################################################
        # # Rename y column to "y" for universality
        # df = df.rename(columns={target_name[0]: "y"})
        # target_name = ["y"]
        # target_series = df[[target_name]].dropna()

        indicator_cols = [c for c in df.columns if c not in target_series]
        indicator_cols = pre_selected_features
        # TODO :  save as pickle file
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

        ########################## start multiplicative features #####################
        # These are appropriate if forecast_type is set to "proportional_diff"
        df = df.join(df[[target] + indicator_cols].ewm(alpha=0.6).mean().pct_change().add_suffix("_ewm_pct_change1"))
        df = df.join(df[[target] + indicator_cols].ewm(alpha=0.6).mean().pct_change(periods=2).add_suffix("_ewm_pct_change2"))
        # df = df.join(df[[target] + indicator_cols].ewm(alpha=0.6).mean().pct_change(periods=3).add_suffix("_ewm_pct_change3"))
        ## setting: add skew
        skew_setting = [6]  # should be at least 3
        df, skewcols = fe.add_skew(df, features=[target] + indicator_cols, window=skew_setting)
        # Trigonometric encoding of months
        df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
        # Drop the original raw indicators and target series to improve performance
        df = df.drop(columns=[target] + indicator_cols)
        #Added crisis dummy
        crisis_dummy = (df.index >= "2008-05-01") & ( df.index < "2008-12-01" )
        covid_dummy = (df.index >= "2020-02-01")  & ( df.index < "2020-12-01" )
        debt_ceiling_2011 = (df.index >= "2011-08-01") & ( df.index < "2012-01-01" )
        debt_ceiling_2013 = (df.index >= "2013-02-01") & ( df.index < "2013-07-01" )
        chip_shortage = (df.index >= "2021-05-01") & ( df.index < "2021-12-01" )
        df['crisis_dummy'] = crisis_dummy | covid_dummy | debt_ceiling_2011 | debt_ceiling_2013 | chip_shortage
        # self.feat_cols.append('crisis_dummy')

        ########################### end multiplicative features #######################
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   TRAIN, PREDICT & HYPERTUNE
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        # CONFIGURE HYPEROPT
        max_hyperopt_evals = 2
        cv = sklearn.model_selection.TimeSeriesSplit(10, test_size=6)  # Cross-validation
        # import pdb;pdb.set_trace()
        df = reduce_mem_usage(df)
        X_train = df.loc[df.index < backtest_start_date]
        #
        # Fit hyperparameters of LightGBM regression forecasters
        # Define the Hyperopt search space
        #
        # space = space_smoothboost(X_train)
        space = {
            # Feature Selector part
            'sel__estimator': {
                "type": "lgb.LGBMRegressor(boosting_type= 'gbdt'," \
                        " colsample_bytree=0.27, learning_rate=0.056," \
                        " max_depth=7, n_estimators=100, num_leaves=16,"
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
            'est__n_estimators':hyperopt.pyll.scope.int(hp.quniform('est__n_estimators', low = 4, high= min(64, int(len(X_train)//2)), q= 1)),
            'est__num_leaves': hyperopt.pyll.scope.int(hp.qloguniform('est__num_leaves', low=np.log(4), high=np.log(64), q=1 )), #max number of leaves in one tree
            'est__min_child_samples':hyperopt.pyll.scope.int(hp.qloguniform('est__min_child_samples',low =np.log(4), high=np.log(2**4+1), q=1)),  
            'est__colsample_bytree':hp.uniform('est__colsample_bytree', low =0.01, high=1.0),
            # 'est__num_leaves': hyperopt.pyll.scope.int(hp.quniform('est__num_leaves', low=10, high=40, q=1)),
            # Setting learning_rate and n_estimators is mainly a matter of how long you're willing to wait,
            # keep learning_rate high & n_estimators low during development for speed then increase for final results
            'est__max_bin':hyperopt.pyll.scope.int(hp.qloguniform('est__max_bin', low =np.log(7),high= np.log(128), q=1)),
            'est__lambda_l1':hp.loguniform('est__lambda_l1', low =np.log(1.0/16),high=np.log(16)),
            'est__lambda_l2':hp.loguniform('est__lambda_l2', low =np.log(1.0/16), high=np.log(16)),
            'est__learning_rate':hp.loguniform('est__learning_rate', low =np.log(0.001), high=np.log(1.0)),
            # 'est__subsample': hp.uniform('est__subsample', low=0.3, high=1.0),
            'est__bagging_fraction': hp.uniform('est__bagging_fraction', low=0.01, high=1), #alias "subsample
            # 'est__subsample_freq': hyperopt.pyll.scope.int(hp.quniform('est__subsample_freq', low=1, high=10, q=1)),
            # 'est__learning_rate': 0.01,
            'est__max_depth':hyperopt.pyll.scope.int(hp.quniform('est__max_depth', low =2, high=32, q=1)),
            # 'est__bagging_fraction': hp.uniform('est__bagging_fraction', low=0.01, high=1), #alias "subsample"
            'est__feature_fraction': hp.uniform('est__feature_fraction', low=0.01, high=1),
            'est__random_state': 42,
            'est__boosting_type': 'dart'
        }
        
        regressor = lgb.LGBMRegressor

        # Fit hyperparams on pre-backtest data
        # TODO can be unselected if hyperparameters are known.
        
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
            meta_features=meta_features
        )
        

        forecast_tables_val, rmses_val, mapes_val, maes_val, wmapes_val = backtest_hyperparams(
            regressor, 
            df, 
            target_series,
            best_hyperparams_list,
            validation_start_date,
            validation_end_date, 
            forecast_type,
            meta_feature_prefix=meta_feature_prefix,
            meta_features=meta_features,
            deploy_mode=False
        )
        
        forecast_tables_train, rmses_train, mapes_train, maes_train, wmapes_train = get_train_results(
            regressor, 
            df.loc[df.index < validation_start_date], 
            target_series, 
            best_hyperparams_list,
            forecast_type=forecast_type, 
            meta_feature_prefix=meta_feature_prefix, 
            meta_features=meta_features,
            deploy_mode=False
        )

        # Get backtest results THIS HAS TO BE AFTER THE TRAIN AND VAL RESULTS SO THE CORRECT RMSES & MAPES ARE LOGGED
        forecast_tables_test, rmses, mapes, maes, wmapes = backtest_hyperparams(
            regressor, 
            df, 
            target_series,
            best_hyperparams_list,
            backtest_start_date,
            backtest_end_date, 
            forecast_type,
            meta_feature_prefix=meta_feature_prefix,
            meta_features=meta_features,
            deploy_mode=False)
        
        values_frame_test = metrics_frame_single(1, "Test", rmses, mapes, maes, wmapes)
        values_frame_val = metrics_frame_single(1, "Validation", rmses_val, mapes_val, maes_val, wmapes_val)
        values_frame_train = metrics_frame_single(1, "Train", rmses_train, mapes_train, maes_train, wmapes_train)
        metrics_frame = pd.concat((values_frame_train, values_frame_val, values_frame_test), axis=0)

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   LOG INFORMATION INT MLFLOW
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        save_residual_frames(forecast_tables_train, forecast_tables_val, forecast_tables_test)
        #
        # path2save_features = os.path.join(base_path, "BASF_Metals", "artifacts", "forecast","pre_selected_features.pkl")
        # pre_selected_features.to_pickle(path2save_features, protocol=4)
        # mlflow.log_artifact(local_path = path2save_features, artifact_path ="forecasts")
        path2save_granger = os.path.join(base_path, "BASF_Metals", "artifacts", "forecast","df_granger.pkl")
        df_granger.to_pickle(path2save_granger, protocol=4)
        mlflow.log_artifact(local_path = path2save_granger, artifact_path ="forecasts")
        path2save_correlation = os.path.join(base_path, "BASF_Metals", "artifacts", "forecast","df_correlation.pkl")
        df_correlation.to_pickle(path2save_correlation, protocol=4)
        mlflow.log_artifact(local_path = path2save_correlation, artifact_path ="forecasts")
        path2save_meta = os.path.join(base_path, "BASF_Metals", "artifacts", "forecast","meta_features.pkl")
        meta_features.to_pickle(path2save_meta, protocol=4)
        mlflow.log_artifact(local_path = path2save_meta, artifact_path ="forecasts")


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
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   CREATE FIGURES
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        # Barplot of backtest results
        # rmse_series = pd.Series(rmses, index=["H1", "H2", "H3", "H4", "H5", "H6"])
        mape_series = pd.Series(mapes, index=["H1", "H2", "H3", "H4", "H5", "H6"])
        fig, ax = plt.subplots(figsize=(5, 8))
        ax = mape_series.plot.bar(rot=0, ax=ax)
        _ = ax.set_title(f"Overall Acc = {np.mean(mapes):.2f}")
        _ = ax.set_ylabel("Error %")
        plt.tight_layout()
        sns.despine()
        # Save figure to outputs directory as mlflow.log_figure() overrides background colour BASF_Metals\Models\notebooks\outputs
        fig.savefig("BASF_Metals/Models/notebooks/outputs/backtest/barplot_rmse_backtest.png", facecolor=fig.get_facecolor(), edgecolor="none")

        # Create and save tables of forecasted values
        for i, table in enumerate(forecast_tables_test):
            table.to_csv(f"BASF_Metals/Models/notebooks/outputs/backtest/backtest_output_table_t{i+1}.csv")

        # Triple barplot of backtest forecasts by horizon
        fig = fu.plot_backtest_forecasts(target_series, forecast_tables_test, figsize=(8, 8))
        fig.savefig("BASF_Metals/Models/notebooks/outputs/backtest/backtest_forecasts_plot.png", facecolor=fig.get_facecolor(), edgecolor="none")
        
        ############ Return forecast
        # Train a multitimestep model on all the data and create the forecast
        # Format: DT index (with all dates?), "forecast", "lower", "upper". Lower and upper optional.
        # Create models with the previously-optimized hyperparameters

        """mlflow.start_run creates a new MLflow run to track the performance of this model. 
        Within the context, you call mlflow.log_param to keep track of the parameters used, and
        mlflow.log_metric to record metrics like accuracy."""
        with mlflow.start_run(run_name='MultipleTimestepRegressionForecaster', nested=True) as MultipleTimestepRegressionForecaster:
            # get current run and experiment id
            runID = MultipleTimestepRegressionForecaster.info.run_id
            experimentID = MultipleTimestepRegressionForecaster.info.experiment_id


            pipeline_list = [pipeline_model.PipelineModel(hps=h, estimator=regressor, multiout_wrapper=False) for h in best_hyperparams_list]
            model = rf.MultipleTimestepRegressionForecaster(
                pipeline_list, horizon, forecast_type,
                meta_feature_prefix=meta_feature_prefix
            )
            # mlflow.sklearn.log_model(sk_model=model,registered_model_name  ="Multiple_Timestep_Regression_Forecaster_unfit", artifact_path = os.path.join( "BASF_Metals", "artifacts","Model"))
            model = model.fit(df, y=target_series, meta_features=meta_features)
            mlflow.sklearn.log_model(sk_model=(model.fit(df, y=target_series, meta_features=meta_features)),registered_model_name  ="Multiple_Timestep_Regression_Forecaster_fit", artifact_path = os.path.join( "BASF_Metals", "artifacts","Model"))
            # mlflow.pyfunc.log_model(python_model=(model.fit(df, y=target_series, meta_features=meta_features)),registered_model_name  ="Multiple_Timestep_Regression_Forecaster_fit", artifact_path = os.path.join( "BASF_Metals", "artifacts","Model"))
            # pickle
            forecast = model.predict(conf_ints=True)
            forecast = pd.concat((target_series, forecast), axis=0).drop(columns=target)
            import pdb;pdb.set_trace()
    # ======================================================================================================================================
            # Log the model with a signature that defines the schema of the model's inputs and outputs. 
            # When the model is deployed, this signature will be used to validate inputs.
            # signature = infer_signature(X_train, model.predict(None, X_train))
            # # MLflow contains utilities to create a conda environment used to serve models.
            # # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
            # conda_env =  _mlflow_conda_env(
            #         additional_conda_deps=None,
            #         additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
            #         additional_conda_channels=None,
            #     )

            # mlflow.sklearn.log_model(model,"Multiple_Timestep_Regression_Forecaster")
            # log model
            forecast = model.predict(conf_ints=True)
            ############ Return feature importances
            ## Train a model on all the data and get the feature importances for the 1 month forecaster?
            # Format: columns "feature" with feature names and "importance" with the absolute values. Set "feature" as the index.
            feature_importances = model.forecasters[0].regressor.get_feature_importance(df.columns, [target])
            feature_importances = feature_importances.rename(columns={"y": "importance"})
            feature_importances.index.name = "feature"

            ############ Return actuals
            # The target series, but index aligned with the forecast (same start date but end date does not extend past present)
            target_series.index.name = "Date"
            target_series = target_series.rename(columns={target: "actual"})

            
            for i in range(0, 3):
                backtesting = forecast_tables_test[0]
                backtesting = target_series.join(backtesting, how="inner")
                backtesting.to_csv(os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", f"backtest_{i}.csv"))
                # mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", f"backtest_{i}.csv"))

            metrics_frame.to_csv(os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "g_metrics_frame.csv"))
            mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "g_metrics_frame.csv"))
            target_series.to_csv(os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "target_series.csv"))
            mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "target_series.csv"))
            forecast.to_csv(os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "forecast.csv"))
            mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "forecast.csv"))
            feature_importances.to_csv(os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "feature_importance.csv"))
            mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", "feature_importance.csv"))

            # Logging training data
            mlflow.log_artifact(local_path = os.path.join(base_path, "BASF_Metals", "Models", "notebooks", "outputs", "results_run", f"backtest_{i}.csv"))
            mlflow.end_run()  # In case the last run was not ended

        