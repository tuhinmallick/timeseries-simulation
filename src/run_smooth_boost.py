# =======================================================================================
#   IMPORT LIBRARIES
# =======================================================================================

import os,sys, pathlib,pickle,traceback
from pickletools import TAKEN_FROM_ARGUMENT1
import sys
import logging
import pickle
import shutil

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core import Workspace
from azureml.core.datastore import Datastore
from azureml.core import Dataset


import mlflow
from mlflow.exceptions import MlflowException
from sqlalchemy import true


# base_path = r"C:\Users\OchsP\Documents\Python-dev\forecasty-lab"
# base_path = r"C:\Users\mallict\forecasty-lab"
src_location = pathlib.Path(__file__).absolute().parent
artifact_location = os.path.join(pathlib.Path(__file__).absolute().parent.parent, 'artifacts')


if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
from utils.main_regression_deploy import *
from utils.reduce_memory_usage import reduce_mem_usage
from utils.grange_and_correlate import grange_and_correlate, user_input_correlation_picker
from utils.remove_missing_data import remove_features_with_na
from utils.smb_feature_engineering import feature_engineering_pipeline

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
    subscription_id = 'fc563dde-a9ad-4edf-9998-2d52ba8afff9'
    # Azure Machine Learning resource group 
    resource_group = 'forecasty_ml_test' 
    # Azure Machine Learning workspace name
    workspace_name = 'forecasty_ml_workplace'
    # Instantiate Azure Machine Learning workspace
    ws = Workspace.get(name=workspace_name,
                    subscription_id=subscription_id,
                    resource_group=resource_group)
    datastore = ws.get_default_datastore()
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
    return datastore
    # dictionary
    # backend_config = {"USE_CONDA": True}
    # local_mlproject_run = mlflow.projects.run(uri=".", 
    #                                 # parameters={"alpha":0.3},
    #                                 backend = "azureml",
    #                                 backend_config = backend_config)
    # print(local_mlproject_run)



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
    #   KEY PARAMETERS AND SETTINGS
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    try:
        target_name = "Platinum"
        target = f"{target_name}_spot_price" # This is used as column selector.
        horizon = 3
        
        forecast_type = "absolute_diff"  # Either "proportional", "absolute_diff" or "absolute"
        backtest_val_periods = 24  # Number of periods (e.g. months) used for backtesting and validation (each)
        validation_split_size = 3  # Number of validation periods in each validation split
        
        HYPEROPT_FMIN_SEED = 123
        os.environ["HYPEROPT_FMIN_SEED"] = str(HYPEROPT_FMIN_SEED)
        
        # Set the exfcel parameters.
        file_name = "PGM_Data_Month"
        sheet_names = ["Data Dictionary", "Compiled Dataset"]
        
        # Configure the hyperopt setting
        max_hyperopt_evals = 30
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #   CREATE TARGET FOLDERS
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        basf_backtest_platinum = {
            "steady": {
                "backtest_start_date": "01-12-2015",
                "backtest_end_date": "01-02-2019"
                },
            "crisis_up": {
                "backtest_start_date": "01-04-2020",
                "backtest_end_date": "01-01-2021"
            },
            "crisis_down": {
                "backtest_start_date": "01-03-2020",
                "backtest_end_date": "01-05-2020"
            }
        }

        basf_backtest_palladium = {
            "steady": {
                "backtest_start_date": "01-12-2015",
                "backtest_end_date": "01-02-2019"
                },
            "crisis_up": {
                "backtest_start_date": "01-02-2021",
                "backtest_end_date": "01-04-2021"
            },
            "crisis_down": {
                "backtest_start_date": "01-02-2020",
                "backtest_end_date": "01-06-2020"
            }
        }

        basf_backtest_rhodium = {
            "steady": {
                "backtest_start_date": "01-12-2015",
                "backtest_end_date": "01-02-2019"
                },
            "crisis_up": {
                "backtest_start_date": "01-11-2019",
                "backtest_end_date": "01-01-2022"
            }
        }
        
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
        datastore = _mlflow(target_name)
        # Timing of parent run starts from here
        with mlflow.start_run(run_name='single_timestep_run') as single_timestep_run : 

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            #   READ AND CONFIGURE THE DATA
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            # =========================== RAW UNTRANSFORMED DATA ===========================
            # Get the base data and adjust.
            df = pd.read_excel(
                os.path.join(artifact_location,'source_data', file_name + ".xlsx"),
                sheet_names[1]
                )
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
            path2save_raw_data = os.path.join(artifact_location,'dataframes',"df.pkl")
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df, target= (datastore,"forecast/dataframe"), name="df_raw", show_progress = True)
            # df.to_pickle(path2save_raw_data, protocol=4)
            mlflow.log_artifact(local_path = path2save_raw_data, artifact_path ="forecasts")

            # Read the data dictionary with the full names.
            df_dict = pd.read_excel(
               os.path.join(artifact_location,'source_data', file_name + ".xlsx"),
                sheet_names[0]
                )[["Feature Name", "Full Name"]]
            path2save_df_dict = os.path.join(artifact_location,'dataframes',"df_dict.pkl") 
            df.to_pickle(path2save_df_dict, protocol=4)
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df, target="forecast/dataframe", name="df_dict")
            mlflow.log_artifact(local_path = path2save_df_dict, artifact_path ="forecasts")
            # Rename the target
            # df.columns = df_dict["Full Name"].values.tolist()
            
            # =========================== TIDY DATA ===========================
            # Remove features that contain a lot of missing data.
            df = remove_features_with_na(df, threshold=65, trailing_value=5)
            df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
            path2save_tidy_df = os.path.join(artifact_location,'dataframes',"df_tidy.pkl")
            df.to_pickle(path2save_tidy_df, protocol=4)
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df, target="forecast/dataframe", name="df_tidy")
            mlflow.log_artifact(local_path = path2save_tidy_df, artifact_path ="forecasts/dataframe")
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            #   SET META FEATURES
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            import pdb;pdb.set_trace()

            meta_features = create_meta_feature_frame(df[target].dropna(), horizon, ci_alpha=0.05)
            meta_feature_prefix = "meta__ETS"
            path2save_meta_features = os.path.join(artifact_location,'dataframes',"meta_features.pkl")
            meta_features.to_pickle(path2save_meta_features, protocol=4)
            # Dataset.Tabular.register_pandas_dataframe(dataframe=meta_features, target="forecast/dataframe", name="meta_features")
            mlflow.log_artifact(local_path = path2save_meta_features, artifact_path ="forecasts")

            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            #   FEATURE ENGINEERING
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            
            # =========================== PREPARE FOR SIMULATION ===========================

            simulation_correlation_dict = user_input_correlation_picker(df[target], df.drop([target], axis=1), selected_method="mean", max_lags=30)
            # Get preselected features and other causality and correlation metrics
            pre_selected_features, df_granger, df_correlation = grange_and_correlate(df, target, granger_lags=12, correlation_lags=12, number_of_features=10)
            # Putiing the indicator columns as pre selected features
            indicator_cols = pre_selected_features
            df = df[pre_selected_features+[target]]
            df, target_series = feature_engineering_pipeline(df=df, target=target, horizon=horizon, pre_selected_features=indicator_cols, forecast_type=forecast_type)

            # The local paths to the dataframes
            path2save_granger = os.path.join(artifact_location,'dataframes',"df_granger.pkl")
            path2save_correlation = os.path.join(artifact_location,'dataframes',"df_correlation.pkl")
            path2save_fe_df = os.path.join(artifact_location,'dataframes',"fe_df.pkl")
            path2save_target = os.path.join(artifact_location,'dataframes',"target_series.pkl")
            path2save_pre_selected_features = os.path.join(artifact_location,'dataframes',"pre_selected_features_df.pkl")

            # Pickling the dataframes
            df_granger.to_pickle(path2save_granger, protocol=4)
            df_correlation.to_pickle(path2save_correlation, protocol=4)
            df.to_pickle(path2save_fe_df, protocol=4)
            df[pre_selected_features].to_pickle(path2save_pre_selected_features, protocol=4)
            target_series.to_pickle(path2save_target, protocol=4)

            # Logging the artifacts in MlFlow
            mlflow.log_dict(simulation_correlation_dict, artifact_file = "simulation/simulation_correlation_dict.json")
            mlflow.log_artifact(local_path = path2save_granger, artifact_path ="forecasts/dataframes")
            mlflow.log_artifact(local_path = path2save_correlation, artifact_path ="forecasts/dataframes")
            mlflow.log_artifact(local_path = path2save_fe_df, artifact_path ="forecasts/dataframes")
            mlflow.log_artifact(local_path = path2save_target, artifact_path ="forecasts/dataframes")
            mlflow.log_artifact(local_path = path2save_pre_selected_features, artifact_path ="forecasts/dataframes")
            
            # Saving to dataset
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df_granger, target="forecast/dataframe", name="df_granger")
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df_correlation, target="forecast/dataframe", name="df_correlation")
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df, target="forecast/dataframe", name="df_fe")
            # Dataset.Tabular.register_pandas_dataframe(dataframe=df[pre_selected_features], target="forecast/dataframe", name="pre_selected_features_df")
            # Dataset.Tabular.register_pandas_dataframe(dataframe=target_series, target="forecast/dataframe", name="target_series")
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            #   BACKTESTING CONFIGURATION
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

            # Set the backtesting periods.
            if backtest_val_periods % validation_split_size:
                raise ValueError("validation_split_size must be a factor of backtest_val_periods")
            backtest_end_date = df.index[-1]
            backtest_start_date = df.index[-1] - df.index.freq * (backtest_val_periods - 1)
            validation_start_date = df.index[-1] - df.index.freq * ((backtest_val_periods * 2) - 1)
            validation_end_date = df.index[-1] - df.index.freq * (backtest_val_periods)
            
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            #   TRAIN, PREDICT & HYPERTUNE
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                
            # NOTE Alternative cv selection.
            cv = sklearn.model_selection.TimeSeriesSplit(
                int(backtest_val_periods / validation_split_size), 
                test_size=int(validation_split_size)
            )
            cv = sklearn.model_selection.TimeSeriesSplit(12, test_size=5)  # Cross-validation
            
            # Reduce Memory Useage of Data Frame
            df = reduce_mem_usage(df)
            X_train = df.loc[df.index < backtest_start_date]
            
            # Define the Hyperopt search space
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
                'sel__step': int(len(X_train.columns)*0.25),
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

            mlflow.log_param("Backtest start", backtest_start_date)
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("forecast_type", forecast_type)
            mlflow.log_param("max_hyperopt_evals", max_hyperopt_evals)
            mlflow.set_tag("Model type", "LightGBM regression forecaster")
            
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
            _index =[]
            for h in range(1, horizon+1):
                _index = _index + [f'H{h}']
            mape_series = pd.Series(mapes, index=_index)
            fig, ax = plt.subplots(figsize=(5, 8))
            ax = mape_series.plot.bar(rot=0, ax=ax)
            _ = ax.set_title(f"Overall Acc = {np.mean(mapes):.2f}")
            _ = ax.set_ylabel("Error %")
            plt.tight_layout()
            sns.despine()
            # Save figure to outputs directory as mlflow.log_figure() overrides background colour BASF_Metals\Models\notebooks\outputs
            fig.savefig("BASF_Metals/Models/notebooks/outputs/backtest/barplot_rmse_backtest.png", facecolor=fig.get_facecolor(), edgecolor="none")
            mlflow.log_figure(fig, "forecasts/backtest/barplot_rmse_backtest.png")

            # Create and save tables of forecasted values
            for i, table in enumerate(forecast_tables_test):
                table.to_csv(f"BASF_Metals/Models/notebooks/outputs/backtest/backtest_output_table_t{i+1}.csv")
                mlflow.log_artifact(local_path = f"BASF_Metals/Models/notebooks/outputs/backtest/backtest_output_table_t{i+1}.csv", artifact_path ="forecasts")

            # Triple barplot of backtest forecasts by horizon
            fig = fu.plot_backtest_forecasts(target_series, forecast_tables_test, figsize=(8, 8))
            fig.savefig("BASF_Metals/Models/notebooks/outputs/backtest/backtest_forecasts_plot.png", facecolor=fig.get_facecolor(), edgecolor="none")
            mlflow.log_figure(fig, "forecasts/backtest/backtest_forecasts_plot.png")
            
            ############ Return forecast
            # Train a multitimestep model on all the data and create the forecast
            """mlflow.start_run creates a new MLflow run to track the performance of this model. 
            Within the context, you call mlflow.log_param to keep track of the parameters used, and
            mlflow.log_metric to record metrics like accuracy."""
            with mlflow.start_run(run_name='MultipleTimestepRegressionForecaster', nested=True) as MultipleTimestepRegressionForecaster:
                pipeline_list = [pipeline_model.PipelineModel(hps=h, estimator=regressor, multiout_wrapper=False) for h in best_hyperparams_list]
                model = rf.MultipleTimestepRegressionForecaster(
                    pipeline_list, horizon, forecast_type,
                    meta_feature_prefix=meta_feature_prefix
                )
                import pdb;pdb.set_trace()
                model_name = f"{target_name}_model_unfit.pkl"
                with open(os.path.join(artifact_location,'model',model_name), 'wb') as _model:
                     pickle.dump(model, _model, pickle.HIGHEST_PROTOCOL)
                mlflow.sklearn.log_model(sk_model=model,registered_model_name  ="Multiple_Timestep_Regression_Forecaster_unfit", artifact_path = os.path.join( "BASF_Metals", "artifacts","Model"))
                model = model.fit(df, y=target_series, meta_features=meta_features)
                # mlflow.sklearn.log_model(sk_model=(model.fit(df, y=target_series, meta_features=meta_features)),registered_model_name  ="Multiple_Timestep_Regression_Forecaster_fit", artifact_path = os.path.join( "BASF_Metals", "artifacts","Model"))
                _model_name = f"{target_name}_model_fit_{horizon}.pkl"
                with open(os.path.join(artifact_location,'model',_model_name), 'wb') as _model:
                     pickle.dump(model, _model, pickle.HIGHEST_PROTOCOL)
                forecast = model.predict(conf_ints=True)
                _forecast_name = f"{target_name}_forecast_horizon_{horizon}.pkl"
                with open(os.path.join(artifact_location,'dataframes',_forecast_name), 'wb') as _forecast:
                     pickle.dump(forecast, _forecast, pickle.HIGHEST_PROTOCOL)
                forecast = pd.concat((target_series, forecast), axis=0).drop(columns=target)
                model_uri = "runs:/{}/sklearn-model".format(MultipleTimestepRegressionForecaster.info.run_id)
                # mv = mlflow.register_model(model_uri, "Multiple_Timestep_Regression_Forecaster_fit")
                # print("Name: {}".format(mv.name))
                # print("Version: {}".format(mv.version))
                ############ Return feature importances
                # Timestep 1
                feature_importances_1 = model.forecasters[0].regressor.get_feature_importance(df.columns, [target])
                feature_importances_1 = feature_importances_1.rename(columns={"y": "importance"})
                feature_importances_1.index.name = "feature"
                # Timestep 2
                feature_importances_2 = model.forecasters[1].regressor.get_feature_importance(df.columns, [target])
                feature_importances_2 = feature_importances_2.rename(columns={"y": "importance"})
                feature_importances_2.index.name = "feature"
                # Timestep 3
                feature_importances_3 = model.forecasters[2].regressor.get_feature_importance(df.columns, [target])
                feature_importances_3 = feature_importances_3.rename(columns={"y": "importance"})
                feature_importances_3.index.name = "feature"

                os.makedirs(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "outputs"))
                os.makedirs(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "backtest"))
                os.makedirs(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "Feature_importance"))
                os.makedirs(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "series"))
                os.makedirs(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "metrics"))
                ############ Return actuals
                # The target series, but index aligned with the forecast (same start date but end date does not extend past present)
                target_series.index.name = "Date"
                target_series = target_series.rename(columns={target: "actual"})
                import pdb;pdb.set_trace()
                for i in range(0, 3):
                    backtesting = forecast_tables_test[i]
                    backtesting = target_series.join(backtesting, how="inner")
                    backtesting.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "backtest", f"backtest_{i}.csv"))

                metrics_frame.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "metrics", "g_metrics_frame.csv"))
                target_series.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "series", "target_series.csv"))
                forecast.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "series", "forecast.csv"))
                feature_importances_1.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "Feature_importance", "feature_importances_1.csv"))
                feature_importances_2.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "Feature_importance", "feature_importances_2.csv"))
                feature_importances_3.to_csv(os.path.join(base_path, "BASF_Metals", "artifacts",MultipleTimestepRegressionForecaster.info.run_id,"forecast", "Feature_importance", "feature_importances_3.csv"))
                mlflow.log_artifacts(local_dir = (os.path.join(artifact_location,'dataframes',MultipleTimestepRegressionForecaster.info.run_id)), artifact_path = 'forecasts/results_run')
                mlflow.end_run(nested=True)  # In case the last run was not ended
        mlflow.end_run()  # In case the last run was not ended
    except MlflowException as err:
        mlflow.end_run()
        print(err)   

    