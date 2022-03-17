
import os,sys, logging, pathlib,pickle,warnings
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.sklearn
import azureml.core
from azureml.core import Workspace, Datastore, Dataset
src_location = pathlib.Path(__file__).absolute().parent
artifact_location = os.path.join(pathlib.Path(__file__).absolute().parent.parent, 'artifacts')
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
# # save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
import utils.eda_base as eda_base

from utils import grange_and_correlate, smb_feature_engineering, plotly_visualization


# grange_and_correlate import grange_and_correlate, user_input_correlation_picker

def model_retrieve():
    # Use MlFlow to retrieve the run that was just completed
    # client = MlflowClient()
    # experiment_name = client.get_experiment_by_name(target_name)
    # experiment_id = experiment_name.experiment_id
    # # To search all known experiments for any MLflow runs created using the Multitimestepmodel model architecture
    # all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
    # runs = client.search_runs(experiment_id, "", order_by=["metrics.rmse DESC"], max_results=1, run_view_type=ViewType.ALL)
    # best_run = runs[0]
    # lookup_metric = "rmse"
    # best_run, fitted_model = remote_run.get_output(metric = lookup_metric)
    # # runs = client.search_runs(experiment_ids=all_experiments, filter_string="params.model = 'Inception'", run_view_type=ViewType.ALL)
    # # lookup_metric = "root_mean_squared_error"
    # # best_run, fitted_model = remote_run.get_output(metric = lookup_metric)

    # # download_artifacts(run_id: str, path: str, dst_path: Optional[str] = None) 
    # metrics = finished_mlflow_run.data.metrics
    # tags = finished_mlflow_run.data.tags
    # params = finished_mlflow_run.data.params

    # print(metrics,tags,params)
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
    # mf.set_env_vars()       # Setting the environment variables
    # mf.create_s3_bucket()   # Create the S3 bucket in MinIO if it doesn't exist
    # registry_uri =  r'sqlite:///C:\Users\mallict\forecasty-lab\BASF_Metals\artifacts\mlflow_db.db' #'sqlite:///mlflow_db.db'
    tracking_uri = ws.get_mlflow_tracking_uri()
    print(tracking_uri)
    # mlflow.tracking.set_registry_uri(registry_uri)
    mlflow.tracking.set_tracking_uri(tracking_uri)
    model_name = "Multiple_Timestep_Regression_Forecaster_fit"
    stage = 'Production'
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    return model




def features_picker(simulation_dict, sim_df, target,original_target, horizon):
    modified_pre_selected_df, simulation_correlation, change_pre_selected_df =None, {}, None
    # subscription_id = 'fc563dde-a9ad-4edf-9998-2d52ba8afff9'
    # # Azure Machine Learning resource group 
    # resource_group = 'forecasty_ml_test' 
    # # Azure Machine Learning workspace name
    # workspace_name = 'forecasty_ml_workplace'
    # # Instantiate Azure Machine Learning workspace
    # ws = Workspace.get(name=workspace_name,
    #                 subscription_id=subscription_id,
    #                 resource_group=resource_group)
    # blob_datastore_name='workspaceblobstore' # Name of the datastore to workspace
    # container_name=os.getenv("BLOB_CONTAINER", "azureml-blobstore-5622c08c-e4fd-4d5a-b2ba-842ccdbea3f4") # Name of Azure blob container
    # account_name=os.getenv("BLOB_ACCOUNTNAME", "forecastymlwor5588590134") # Storage account name
    # account_key=os.getenv("BLOB_ACCOUNT_KEY", "OjZI0eJQm1THrk10yVIpWTBYEXI3UMairzfQnYLjjZ7/+5YYKlVW/FZVtG5uhGi/oR0WSnXe9FmXVF0r+zuGMA==") # Storage account access key

    # # blob_datastore = Datastore.register_azure_blob_container(workspace=ws, 
    # #                                                         datastore_name=blob_datastore_name, 
    # #                                                         container_name=container_name, 
    # #                                                         account_name=account_name,
    # #                                                         account_key=account_key)

    # datastore = Datastore.get(workspace=ws, datastore_name=blob_datastore_name)
    dataframe_path=os.path.join(artifact_location,'dataframes', original_target, f'horizon_{horizon}')
    meta_features_name = f"{original_target}_meta_features_{horizon}.pkl"
    path2save_meta_features = os.path.join(dataframe_path, meta_features_name)
    with open(path2save_meta_features, 'rb') as meta_features:
        meta_features_df = pickle.load(meta_features)
    pre_selected_features_df_name = f"{original_target}_pre_selected_features_df_{horizon}.pkl"
    path2save_pre_selected_features = os.path.join(dataframe_path,pre_selected_features_df_name)
    with open(path2save_pre_selected_features, 'rb') as pre_selected_features:
        pre_selected_features_df = pickle.load(pre_selected_features)
    target_series_name = f"{original_target}_target_series_{horizon}.pkl"
    path2save_target = os.path.join(dataframe_path,target_series_name)
    with open(path2save_target, 'rb') as target_series:
        target_series_df = pickle.load(target_series)
    first_iteration = False
    # To get the correlation values as dictionary
    print(pre_selected_features_df)
    print(simulation_dict)
    pre_selected_features_columns = [c for c in pre_selected_features_df.columns]
    # pre_selected_features_df = pd.concat([pre_selected_features_df, target_series_df], axis=1)
    # print("==============================================Feature engineering Starting =================================================================================")
    # pre_selected_features_df, _ = smb_feature_engineering.feature_engineering_pipeline(pre_selected_features_df, target, horizon, pre_selected_features_columns, forecast_type="absolute_diff")
    # print("==============================================Feature engineering Done =================================================================================")
    for key, value in simulation_dict.items():
        simulation_correlation = grange_and_correlate.user_input_correlation_picker(sim_df[key], pre_selected_features_df, selected_method='max', max_lags=12)
        simulation_correlation_df = pd.DataFrame(simulation_correlation, index=[1]).melt()
        simulation_correlation_df.columns = ["Feature Name", "Correlation fraction for {}".format(key)]
        if first_iteration:
            final_simulation_correlation_df["Correlation fraction for {}".format(key)] = simulation_correlation_df[ "Correlation fraction for {}".format(key)]
        else : 
            final_simulation_correlation_df = simulation_correlation_df
        change_pre_selected_df= pre_selected_features_df.copy()
        modified_pre_selected_df = pre_selected_features_df.copy()
        for feature_name, correlation in simulation_correlation.items():
                change_pre_selected_df[feature_name]= pre_selected_features_df[feature_name] * correlation * (value/100)
                modified_pre_selected_df[feature_name]= pre_selected_features_df[feature_name] + change_pre_selected_df[feature_name]
        # modified_pre_selected_df = modified_pre_selected_df.append(target_series_df, ignore_index=False)
        modified_pre_selected_df = pd.concat([modified_pre_selected_df, target_series_df], axis=1)
        first_iteration = True
    print(modified_pre_selected_df)
    print("==============================================Feature engineering Starting =================================================================================")
    modified_pre_selected_df, _ = smb_feature_engineering.feature_engineering_pipeline(modified_pre_selected_df, target, horizon, pre_selected_features_columns, forecast_type="absolute_diff")
    print("==============================================Feature engineering Done =================================================================================")
    return modified_pre_selected_df, meta_features_df, target_series_df, final_simulation_correlation_df

def load_model( horizon, target, local=True):
    if local:
        # load the model from disk
        _model_name = f"{target}_model_fit_{horizon}.pkl"
        with open(os.path.join(artifact_location,'model',target, f'horizon_{horizon}', _model_name), 'rb') as meta_features:
            loaded_model = pickle.load(meta_features)
    else :
        pass
    return loaded_model


def main(simulation_dict : dict, sim_df : pd.DataFrame, target: str, horizon: int):
    """
    sim_target : The feature you want to change"""
    target_name = f"{target}_spot_price"
    print(simulation_dict)
    modified_pre_selected_df, meta_features_df, target_series, final_simulation_correlation_df = features_picker(  sim_df= sim_df, simulation_dict = simulation_dict,target=target_name,original_target =target, horizon=horizon)
    # print(modified_pre_selected_df['EXR_CNY_USD'])
    model = load_model(local=True, horizon= horizon, target=target)
    # model = model.fit(modified_pre_selected_df, y=target_series, meta_features=meta_features_df)
    forecast = model.predict( X=modified_pre_selected_df.iloc[-1:], meta_features=meta_features_df, conf_ints=True)
    # forecast = pd.concat((target_series, forecast), axis=0).drop(columns=target)
    print (forecast)
    _forecast_name = f"{target}_forecast_horizon_{horizon}.pkl"
    with open(os.path.join(artifact_location,'dataframes',target, f'horizon_{horizon}',_forecast_name), 'rb') as _fe_df:
        original_forecast= pickle.load(_fe_df)
    forecast_fig = plotly_visualization.plotly_plot_simulation( df_actuals = target_series.rename(columns= {target_name:'actual'}),df_forecast=original_forecast,  df_simulation = forecast,horizon=horizon, display_fig=False, figsize=(1200, 600))

    print("==============================================Simulation Done =================================================================================")
    return forecast_fig, forecast, original_forecast, final_simulation_correlation_df
    
if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore")
        # # Setting the seed 
        # np.random.seed(40)
        # commodity = 'CLI_CHN'
        # perc_change = 10
        # simulation_dict = dict([(commodity,perc_change)])
        # file_name = "Platinum_Drivers_Month"
        # sheet_names = ["Data Dictionary", "Compiled Dataset"]
        # # base_path = r"C:\Users\OchsP\Documents\Python-dev\forecasty-lab"
        # base_path = r"C:\Users\mallict\forecasty-lab"

        # df = pd.read_excel(
        #         os.path.join(base_path, "BASF_Metals", "raw_data", file_name + ".xlsx"),
        #         sheet_names[1]
        #         )
        # target = 'Palladium_spot_price'
        # main(simulation_dict=simulation_dict,sim_df=df, target = target, horizon=3 )
        main()
    except Exception as err:
        logger.exception( 'Error: %s',err)
    pass
