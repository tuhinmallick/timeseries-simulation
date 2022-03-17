#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename : run.py
# Date : 2022-01-20
# Module: commodity-desk-modeling
# Author : Erika Fonseca


import sys, argparse, os
import pathlib
import numpy as np
import pandas as pd

dir = pathlib.Path(__file__).absolute()
sys.path.append(str(dir.parent.parent.parent))
script_path = pathlib.Path(__file__).parent

from fai_io import Config, DataReader
from utils import _logger as logger
from utils._standardize_output import generate_std_output
from models.GradientBoosting._gb import SingleForecast
import pandas as pd

import functools
import logging
import os
import pickle
import shutil
import sys

import hyperopt
import hyperopt.hp as hp
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from tqdm import tqdm

# Import custom libraries
if os.path.realpath("pipelib") not in sys.path:
    sys.path.append(os.path.realpath("pipelib"))
from models.SmoothBoost.pipelib import forecastutils as fu
from models.SmoothBoost.pipelib import pipeline_model
from models.SmoothBoost.pipelib import regressionforecasters as rf
from models.SmoothBoost.pipelib import _feature_engineering as fe
from models.SmoothBoost.pipelib.make_meta_features import create_meta_feature_frame

_logger = logger.config(handler='stdout')

import main_regression_deploy
# from main_regression_deploy import *

def main():
    '''
        This setting consider that we can run the model either passing some arguments on the command line or using environment
        variables
    '''

    # Reading a parameters file generated by commodity desk orchestration service that gathers all the information required
    # to run a customized model
    model_settings = Config(params_dir=params["params_path"]).load()
    params.update(model_settings)

    # The input file is saved with the name of the target by commodity desk orchestration service and the date column
    # is standardize with the name Date
    reader = DataReader(path=params['combined_data_path'])
    # df = reader.read_simple_csv(file_name=params['target'] + '.csv')
    df = pd.read_csv(os.path.join(params['combined_data_path'],params['target']+'.csv'), index_col=0, parse_dates=True)

    # --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # TODO. Implement your model call here
    # --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    df.index.freq = params["frequency"]
    actual, forecast, feature_importance, backtesting = main_regression_deploy.main(df, params["target"], params["horizon"])


    # --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # TODO. Generate the final results following the definitions for the utils functions generate_std_output
    # --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # actual = pd.Series() # TODO. put here the actual
    # forecast= pd.DataFrame() #TODO. put here the forecast and the confidence interval
    # feature_importance = pd.DataFrame() #TODO. put here the feature importance
    # backtesting = pd.DataFrame()  #TODO. put here the actual, forecast and the confidence interval for the backtesting period
    generate_std_output(actual=actual, forecast=forecast, feature_importance=feature_importance,
                        backtesting=backtesting, params=params)


if __name__ == '__main__':
    """
    """
    try:
        params = {}
        parser = argparse.ArgumentParser(description="CommodityForecasting.")

        if "PARAM_PATH" in os.environ:
            interface = "environment variable"
            # Get environment variables
            params["params_path"] = os.environ["PARAM_PATH"]

        else:
            interface = "command-line"
            parser.add_argument("--params_path", required=True)

        args = parser.parse_args()

        if interface == "command-line":
            params = args.__dict__

        main()

    except Exception as e:
        logger.error(_logger, str(e), exc_info=True)
