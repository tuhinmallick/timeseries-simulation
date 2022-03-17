"""
This file is a copy from the same file name in commodity-desk-modeling repo. 
For the up-to-date STLModel please always check the file in the original repo. 
"""

# imports
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR 
from sklearn.linear_model import Lasso, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE

from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
import lightgbm as lgb 

from statsmodels.tsa.seasonal import STL

from matplotlib import pyplot as plt 
from functools import partial 
import os 
import numpy as np 
import pandas as pd
import re  
import logging 
import warnings
import shap 

# from fbprophet import Prophet
# from fbprophet.diagnostics import cross_validation
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.plot import add_changepoints_to_plot 
# from fbprophet.plot import plot_cross_validation_metric 

# TODO: take care of the boundary problem when collecting the trend/season/arma gt. 
# TODO: add back testing method (train till t and use final point to predict t+1~t+h and compare to the ground truth, then do this repeatedly in a rolling window sense. Finally, average t) to check model performance.

# # self defined module
from models.SmoothBoost.pipelib.kaisutils.metrics import MetricsCls
from models.SmoothBoost.pipelib.kaisutils.feature_engineering import create_lead
from models.SmoothBoost.pipelib.kaisutils.misc import config_parser

# turn off INFO message print from target logger
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.WARN)

class suppress_stdout_stderr(object):
    '''
    Ref: https://www.codeforests.com/2020/11/05/python-suppress-stdout-and-stderr/
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



class STLModel(): 
    
    # will be used as model dictionary keys and data dictionary prefix
    mystl_name = pd.Index(['trend', 'season', 'arma'])
    
    def __init__(self, horizon,  model_config={}, ensemble=False, seed=42):
        """
        :horizon: forecast horizon. 
        :model_config: the hyperparams for each part of the model. If ={}, set to the default setting shown in code.
        :ensemble: bool - create the 4th part of the modele to learn 3 weights for 3 prediction parts to form the final prediction. Not implemented yet. Defaults to False. 
        :seed: random seed. Defaults to 42.        
        """
        # args setting
        self.horizon = horizon
        self.seed = seed 
        self.model_config={}
        if model_config=={}: 
            self.model_config = { 
                "trend": {
                    
                    'sel__estimator': hp.choice('trend_sel__estimator', [Lasso(), Ridge()]),
                    'sel__n_features_to_select': ho_scope.int(hp.quniform('trend_sel__n_features', low=10, high=60, q=5)), 
                    'sel__step': hp.choice('trend_sel__step', [10]), 

                    # For RF
                    'est__n_estimators': ho_scope.int(hp.quniform('trend_n_estimators', low=100, high=500, q=25)), 
                    'est__max_features': hp.uniform('trend_max_features', low=0.25, high=0.75), 
                    'est__min_samples_split': hp.uniform('trend_min_samples_split', low=0.05, high=0.2), 
                    'est__max_depth': ho_scope.int(hp.quniform('trend_max_depth', low=2, high=9, q=1 )), 
                    'est__random_state': self.seed
                    # For XGB
                    # 'est__n_estimators': ho_scope.int(hp.quniform('season_n_estimators', low=100, high=500, q=25)), 
                    # 'est__colsample_bytree': hp.uniform('season_colsample_bytree', low=0.25, high=0.75), 
                    # 'est__learning_rate': hp.loguniform('season_learning_rate', low=-5*np.log(10), high=0.5), 
                    # 'est__max_depth': ho_scope.int(hp.quniform('season_max_depth', low=2, high=9, q=1 )), 
                    # For Prophet: always univariate setting, so not good to combine with Pipeline
                    # 'changepoint_range': hp.uniform('trend_changepoint_range', low=.7, high=.95), 
                    # 'changepoint_prior_scale': hp.uniform('trend_changepoint_prior_scale', low=0.01, high=0.25), 
                    # 'seasonality_prior_scale': hp.uniform('trend_seasonality_prior_scale', low=0.5, high=2.5), 
                    # 'holidays_prior_scale': hp.quniform('trend_holidays_prior_scale', low=1, high=10, q=2), 
                }, 
                "season": {

                    'sel__estimator': hp.choice('season_sel__estimator', [Lasso(), Ridge()]),
                    'sel__n_features_to_select': ho_scope.int(hp.quniform('season_sel__n_features', low=10, high=60, q=5)), 
                    'sel__step': hp.choice('season_sel__step', [10]), 
                    # For lgb
                    'est__n_estimators': ho_scope.int(hp.quniform('season_n_estimators', low=100, high=500, q=25)), 
                    'est__colsample_bytree': hp.uniform('season_colsample_bytree', low=0.25, high=0.75), 
                    'est__learning_rate': hp.loguniform('season_learning_rate', low=-5*np.log(10), high=0.5), 
                    'est__max_depth': ho_scope.int(hp.quniform('season_max_depth', low=2, high=9, q=1 )), 
                    'est__random_state': self.seed 
                }, 
                "arma": {

                    'sel__estimator': hp.choice('arma_sel__estimator', [Lasso(), Ridge()]),
                    'sel__n_features_to_select': ho_scope.int(hp.quniform('arma_sel__n_features', low=10, high=60, q=5)), 
                    'sel__step': hp.choice('arma_sel__step', [10]),

                    # For LGBM
                    # 'est__n_estimators': ho_scope.int(hp.quniform('arma_n_estimators', low=100, high=500, q=25)), 
                    # 'est__colsample_bytree': hp.uniform('arma_colsample_bytree', low=0.25, high=0.75), 
                    # 'est__num_leaves': ho_scope.int(hp.quniform('arma_num_leaves', low=5, high=31, q=1 )),
                    # 'est__learning_rate': hp.loguniform('arma_learning_rate', low=-5*np.log(10), high=0.5), 
                    # 'est__max_depth': ho_scope.int(hp.quniform('arma_max_depth', low=2, high=9, q=1 )), 
                    # FOR RF
                    'est__n_estimators': ho_scope.int(hp.quniform('arma_n_estimators', low=100, high=500, q=25)), 
                    'est__max_features': hp.uniform('arma_max_features', low=0.25, high=0.75), 
                    'est__min_samples_split': hp.uniform('arma_min_samples_split', low=0.05, high=0.2), 
                    'est__max_depth': ho_scope.int(hp.quniform('arma_max_depth', low=2, high=9, q=1 )), 
                    'est__random_state': self.seed 
                }
            }
        self.ensemble = ensemble
    
    def __len__(self):
        """return the length of ``self.model``"""
        if not hasattr(self, 'model'): 
            raise AttributeError("attribute `model` has not been built. Please use the fit function firstly.")
        return len(self.model)
    
    @property
    def model(self): 
        if not hasattr(self, '_model'):
            raise AttributeError("attribute `model` has not been built. Please use the fit function firstly.")
        return self._model 
    
    @model.setter
    def model(self, model):
        assert isinstance(model, dict)
        self._model = model
        
    @property
    def horizon(self): 
        return self._horizon
    
    @horizon.setter
    def horizon(self, horizon):
        assert horizon >= 1
        self._horizon = horizon

    def process_data(self, data: pd.DataFrame, targets=None, test_size=.2, freq=None, stl_window_strategy='rolling', 
                     rolling=24, stl_extract_strategy='nanmean', gen_final_strategy='final_pt'):
        """
        process data to the desired form specific to the n-step-ahead STLModel model 
        where n is set up beforehand by the ``horizon`` arg of ``__init__`` method.
        
        :data: orignal dataframe including the target feature(s), the input data should 
            have the Date as in the index and has frequency with it. So data that is not 
            with equally spaced timestamp will output error. This is the limitation of STL method.
        :targets: str or iterable - the name of targets 
        :test_size: float or int - the ratio or size of the test set
        :stl_window_strategy: str in ['rolling', 'expand'] - this decides the window of the seasonal_decompose method will be either rolling or expanding. 
        :rolling: int - the rolling window size for ``seasonal_decompose``. 
            This argument becomes the initial window size if ``stl_strategy`` is 'expand'.
        :stl_extract_strategy: str in ['nanmean', 'last'] - 
            if nanmean, then take the np.nanmean of the overlapping decomposed STL values due to rolling windows.
            if last, then only append the last newly calculated decomposed value in the next rolling window to form the final STL values.
        :gen_final_strategy: str in ['final_pt', 'final_at'] - 
            if final_pt, then use final record of data to generate horizon-ahead-prediction via various dummies fomr 1 up to horizon. 
            if single_at, then use last #=horizon records to generate horizon-ahead-prediction via the last possible unique dummy value.
        """

        # initial check 
        assert stl_window_strategy in ['rolling', 'expand'], "argument `stl_window_strategy` not valid"
        assert stl_extract_strategy in ['nanmean', 'last'], "argument `stl_extract_strategy` not valid"
        assert gen_final_strategy in ['final_pt', 'final_at'], "argument `gen_final_strategy` not valid"
        mydata = data.copy()
        # make target like a list
        if isinstance(targets, str):
            targets = pd.Index([targets])
        elif targets is None: 
            targets = mydata.columns[-1]
        # tranform test_size from ratio(float) to int
        if test_size <= 0: 
            raise ValueError("test size cannot be smaller or equal to zero")
        elif test_size < 1: # case float 
            test_size = int(len(mydata)*test_size)
        elif test_size > len(mydata):
            raise ValueError("test size exceeds the totoal length of input data")
        # fetch self info
        horizon = self.horizon
        # create final dict for info storage
        di = {}
        
        # STL method: add STL decomposed features of the target to data
        mydata = self._make_STL(mydata, targets, freq, stl_window_strategy, rolling, stl_extract_strategy)
        cols = [] 
        for la in targets: 
            cols.extend(  la+'_'+self.mystl_name  )
        di['STL'] = pd.concat([mydata[targets], mydata[cols]], axis=1)
        assert di['STL'].shape[1] == len(targets) * (1 + 3), "#columns is not aligned with #targets*3+1, please check if you have generated the season/trend/arma features for the target(s). If yes, then delete them and try again since STLModel will auto generate these features for target(s). "
        # add features: shut this down if you wanna add the features by yourself
        mydata["year"] = mydata.index.year
        mydata["month"] = mydata.index.month
        mydata["month_sin"] = np.sin(mydata.index.month * 2 * np.pi / 12.0)
        mydata["month_cos"] = np.cos(mydata.index.month * 2 * np.pi / 12.0)
        di['data_before_meta'] = mydata.copy()
        # create lead, add dummy, duplicate data by horizon times, then flatten it
        metadata = self._make_matadata(mydata, targets)
        # sort values 
        metadata = metadata.reset_index().sort_values(by=["DATE", "dummy"]).set_index("DATE")

        di['STL_lead'] = metadata.filter(regex="$|".join(self.mystl_name+'_lead')+'$|^dummy$').copy()
        assert di['STL_lead'].shape[1] == len(targets) * 3 + 1
        
        # store X/y - tra/tes
        ## tra/tes
        tes_dates = metadata.index.unique()[-test_size:]
        tes_mask = metadata.index.isin(tes_dates)
        tra, tes = metadata[~tes_mask], metadata[tes_mask]
        assert tra.shape[0] + tes.shape[0] == metadata.shape[0], "Internal error when partitioning "+\
            f"tra and tes. Size {len(tra)} + {len(tes)} != {len(metadata)}."
        ## store X: always takes awyas trend/season/arma leads from X and 
        ## put only one of them to y
        regex =  '|'.join('_'+ self.mystl_name + '_lead$')
        target_list = metadata.filter(regex=regex).columns
        di['X_tra'] = tra.drop(columns=target_list)
        di['X_tes'] = tes.drop(columns=target_list)
        di['X'] =tra.append(tes).drop(columns=target_list)
        ## store y
        for idx, case in enumerate(self.mystl_name):
            regex = '|'.join(targets+f"_{case}_lead$")
            mask = target_list.str.contains(regex)
            y_tra, y_tes = tra[ target_list[mask] ], tes[ target_list[mask] ]
            y_tra, y_tes = pd.DataFrame(y_tra), pd.DataFrame(y_tes)
            di[case + '_y_tra'] = y_tra
            di[case + '_y_tes'] = y_tes
            di[case + '_y'] = y_tra.append(y_tes)
            
        ## create final X for out-of-data period prediction
        col_order = di["X_tra"].columns       # get col order
        if gen_final_strategy == 'final_pt':
            final = metadata[col_order].iloc[-horizon:, :]
        elif gen_final_strategy == 'final_at':
            mask = metadata.index.unique()[-horizon:]
            mask = metadata.index.isin(mask)
            final =  metadata[col_order][mask][ metadata[mask].dummy==horizon ]
        di['X_final'] = final 
        
        ## store basic info 
        basic_info = {
            'horizon': self.horizon, 
            'targets': targets,
            'target_freq': freq, 
            'train_size': len(mydata) - test_size,  
            'test_size': test_size,
            'test_dates': tes_dates
        }
        di['info'] = basic_info
        
        ## Optional: special case if use Prophet model on, say, trend part
        whichpart = 'trend'
        tmp = di['data_before_meta'].reset_index().rename(
            columns={'index':'ds', targets[0] + f'_{whichpart}': 'y'})
        ### in order to align the dates in other parts: we cut additional horizon 
        di['prophet_tra'] = tmp.iloc[:-test_size-self.horizon] 
        ### note that the tes here has horizon more size than the other test since prophet does contemporaneous pred
        ### in my case I shut down all the other exogenous features.
        di['prophet_tes'] = tmp.iloc[-test_size-self.horizon:] 
        
        return di
    
    def _make_STL(self, data: pd.DataFrame, target: list, freq=None, rolling_strategy='rolling', rolling=24, extract_strategy='nanmean', **kwargs): 
        """
        This function is called by process_data() to generate the 3 features of the target: trend, season, arma. 
        :target: the target feature that will be decomposed by the STL method. 
        :test_size: see process_data() args. 
        :rolling_strategy: str - see process_data() the ``stl_strategy`` argument.
        :rolling: int - the rolling window over seasonal_decompose() to generate the 3 parts.
        :kwargs: dict - additional argument setting for STL method
        :extract_strategy: str - see process_data() the ``stl_extract_strategy`` argument.
        """
        #!!!: core algorithm here to play around 1. with the setting of STL(), and 2.
        # with the strategy of collecting decomposed values.

        # Insight: 1. both STL & seasonal_decompose are very sensitive to `period`.
        # 2: decomposed values in the boundary of rolling window is more error-prone since no data at the opposite side of the current evaluated point.
        # 3. refer to _mask_STL_insight.ipynb to see why I use nanmean method to collect the groun truth decomposed values (welcome to improve this part)

        # freq
        if freq is None: 
            freq = pd.infer_freq(data.index)
        data.index.freq = freq
        # stl method
        final = []
        for tar in target: 
            subli = []
            # case for initial window
            period = STL(data[tar]).config['period']
            seasonal = period + (period % 2 == 0) # Ensure being odd
            obj = STL(data[tar].iloc[:rolling], seasonal=seasonal, **kwargs).fit()
            obj = pd.DataFrame(dict(zip(self.mystl_name, [obj.trend, obj.seasonal, obj.resid])))
            subli.append(obj)
            # case starting the rolling/expanding window 
            for i in range(1, len(data)-rolling+1 ): 
                if rolling_strategy == 'rolling': 
                    start = i 
                elif rolling_strategy == 'expand': 
                    start = 0
                tmp = STL(data[tar].iloc[start:i+rolling], seasonal=seasonal, **kwargs).fit()
                subli.append(
                    pd.DataFrame(  
                        dict(zip(self.mystl_name, [tmp.trend, tmp.seasonal, tmp.resid]))   
                    )
                )
            # compile final df for the current target 
            collector = [] 
            for key in self.mystl_name: # trend/season/arma
                tmpconcated = pd.concat( [subdf[key] for subdf in subli], axis=1 )
                if extract_strategy == 'nanmean':
                    collector.append( np.nanmean( 
                        tmpconcated, 
                        axis=1 
                    ))
                elif extract_strategy == 'last':
                    collector.append( 
                        tmpconcated.bfill(axis=1).iloc[:,0]
                    )
            collector = np.array(collector).T       # after .T => sample_size x 3 
            collector = pd.DataFrame(collector, index=data.index, columns= tar + '_' + self.mystl_name)
            final.append(collector)
        # add back with original data columns 
        output = pd.concat([data, *final], axis=1)
        return output 
    
    def _make_matadata(self, data: pd.DataFrame, target: list):
        """
        Called by process_data() to make input data to be a format ready for n-step-ahead training 
        for each part of the model following Erika's idea. Logic is given below: 
        1. duplicate the original dataset size by horizon times, called tmpdata.
        2. create tmpdummy from 1 to horizon in hope to inform the model that which forecast horizon
            we are predicting for the current record.
        3. create tmptarget from 1 to horizon steps-ahead as our y_train or y_test. Note, if dummy is i, then 
            the corresponding ground truth should also be i-step-ahead.
        4. concat the data in step 1-3 to form the final metadata
        """
        
        horizon = self.horizon
        ## create lead and add dummy 
        cols = [] 
        for la in target: 
            cols.extend(  la+'_'+self.mystl_name  )
        stl_target = data[cols].columns
        tmp = create_lead(data, features=stl_target, lead=range(1, horizon+1))
        for i in range(horizon):
            tmp[f"dummy_{i+1}"] = np.zeros(len(tmp)) + i+1

        ## prepare expanded/flattened features
        tmptarget = pd.concat(
            [tmp.filter(regex=f"_lead{i}$").rename(
                columns= lambda x: re.sub(f'{i}$', '', x)
            ) for i in range(1, horizon+1)], 
            axis=0
        )
        tmpdummy = pd.concat([tmp[i] for i in tmp.filter(regex="dummy_\d+$").columns], axis=0).rename('dummy')
        tmpdata = pd.concat( [data[:-horizon]] * horizon, axis=0)
        assert len(tmpdata) == len(tmpdummy) and len(tmpdummy) == len(tmptarget)
        
        ## combine to get final data and add features
        metadata = pd.concat([tmpdata, tmpdummy, tmptarget], axis=1)
        metadata.rename(columns={0:"dummy", 1:"target"}, inplace=True)
        assert (len(data) - horizon) * horizon == len(metadata)
        
        return metadata
    
    @staticmethod
    def di_checker(di, prefix=''):
        """
        unfold dict and print the shape of the stored values inside. 
        """
        # define sub layer format
        sub_prefix = '|----'
        # pretty print like a tree structure
        for key in di.keys(): 
            if isinstance(di[key], dict): 
                print(prefix+"Key: {} is a dictionary, going to the next layer".format(key))
                STLModel.di_checker(di[key], prefix+sub_prefix)
            else:
                tmplen = len(di[key]) if isinstance(di[key], 
                    (list, tuple, pd.Index, pd.DataFrame, pd.Series, np.ndarray)) else ""
                print(prefix+"Key: {:20s} Value: length {:<20} type {:<40}".format(
                    key, tmplen, str(type(di[key]))), end='')
                if isinstance(di[key], (list, tuple)): 
                    print("values shapes:", [k.shape for k in di[key]])
                elif isinstance(di[key], (int, float, str)): 
                    print("values       :", di[key])
                else: 
                    print("values shapes:", di[key].shape)
        return None
    
    @staticmethod 
    def trend_model(hps): 
        # Note: use MultiOutputRegressor to wrap estimator when the model per se doesn't support multivariate
        sel_hps = config_parser('sel', hps)
        est_hps = config_parser('est', hps)
        return Pipeline([
            ('sca', StandardScaler()), 
            ('sel', RFE(**sel_hps)), 
            ('est', RandomForestRegressor(**est_hps))
        ])
    
    @staticmethod
    def season_model(hps): 
        # Note: use MultiOutputRegressor to wrap estimator when the model per se doesn't support multivariate
        sel_hps = config_parser('sel', hps)
        est_hps = config_parser('est', hps)
        return Pipeline([
            ('sca', StandardScaler()), 
            ('sel', RFE(**sel_hps)), 
            ('est', MultiOutputRegressor(lgb.LGBMRegressor(**est_hps)))
        ])
    
    @staticmethod
    def arma_model(hps):
        # Note: use MultiOutputRegressor to wrap estimator when the model per se doesn't support multivariate
        sel_hps = config_parser('sel', hps)
        est_hps = config_parser('est', hps)
        return Pipeline([
            ('sca', StandardScaler()), 
            ('sel', RFE(**sel_hps)), 
            ('est', RandomForestRegressor(**est_hps))
        ])
    
    @staticmethod
    def meta_model(meta_hps, ensemble=False):
        """
        Called by fit function. Construct trend/season/arma part model and optionally the ensemble model.
        """

        # trend prediction part: individual setting
        trend_hps = meta_hps["trend"]
        season_hps = meta_hps["season"]
        arma_hps = meta_hps["arma"]

        model = {
            "trend": STLModel.trend_model(trend_hps), 
            "season": STLModel.season_model(season_hps), 
            "arma": STLModel.arma_model(arma_hps)
        }
        if ensemble: 
            ensemble_hps = meta_hps["ensemble"]
            # model["ensemble"] = False # to be implemented 
            raise NotImplementedError("not yet allowed for ensemble=True")
        return model 
    
    @classmethod
    def single_objective(cls, hps: dict, case: str, di: dict, cv, scoring='neg_mean_squared_error'):
        """
        Called by fit function. 
        case: determine that trend/season/arma model is the objective.
        PS: One can also implment unified objective() which calls this single function 3 times for training.
        """
        #TODO: I am lazy to align the metric with scoring in the code body
        assert case in cls.mystl_name
        model = eval(f'cls.{case}_model(hps)')
        
        # # case prophet model
        # if isinstance(model, Prophet):
        #     model.add_country_holidays(country_name='US')
        #     #for col in dftra.columns.drop(['ds', 'y']):
        #     #    model.add_regressor(col)
        #     dftra = di['prophet_tra']
        #     with suppress_stdout_stderr():
        #         model.fit(dftra)
        #         # please self change the freq unit w.r.t days.
        #         # Eg, 30 here means we have monthly data
        #         df_cv = cross_validation(
        #             model,
        #             initial=f"{di['info']['train_size'] * 30 // cv.n_splits} days",
        #             period=f"{di['info']['test_size']*30} days",
        #             horizon=f"{di['info']['horizon']} days",
        #             parallel='processes'
        #         )
        #     return performance_metrics(df_cv, rolling_window=1)['rmse'].values[0]
        
        # case the other models
        cv_res = cross_val_score(
            model, X=di[f'X_tra'], y=di[f'{case}_y_tra'], 
            scoring=scoring, cv=cv, n_jobs=-1
        )
        return -cv_res.mean()
    
    def fit(self, di, on='tra', max_evals=50):
        """
        Fit model on either train or on the entire dataset (the latter one is for out-of-data prediction).
        
        :di: dict - the output that is generated by process_data(). 
        :on: str in ['tra', 'all'] - on which data are we fitting the model.
        
        The logic should be: 
        1. call fit with di on train. This will find out the best hyperparams and save them internally.
        2. call fit with di on all. This will take the best hyper that was stored in step 1 to fit the model on all data.
        Note, unless you provide the hyperparms to the object in advance, directly calling the fit on all will output error.
        """
        
        # initial set up 
        assert on in ['tra', 'all'], 'argument on value not allowed.'
        
        # case on==all: no hyper tuning but just use besthyper to refit model 
        if on == 'all':
            self._validate_model()
            for case in self.mystl_name:
                # if isinstance(self.model[case], Prophet):
                #     self.model[case] = Prophet(**self.best_hyper[case])
                #     with suppress_stdout_stderr():
                #         self.model[case].fit(di['prophet_tra'].append(di['prophet_tes']))
                # else:
                self.model[case].fit(di[f'X'], di[f'{case}_y'])
            return self 
            
        # case on==tra
        # find the best hyper pts: train each part instead of all the models, but one can implment so.
        space = self.model_config
        tscv = TimeSeriesSplit(n_splits=5)
        trials = []
        for i in range(len(self.mystl_name)): 
            trials.append(Trials())
        # note: if we do [Trials()] * 3 instead, they actually point to same memory.
        
        best_hyper = {}
        # TODO: multiprocessing, ensemble method
        for i, case in enumerate(self.mystl_name):
            print(f'start the STLModel training of {case} part')
            case_best_pt = fmin(
                fn=partial(self.single_objective, case=case, 
                           di=di, cv=tscv, scoring='neg_mean_squared_error'), 
                space=space[case], 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials[i], 
                rstate=np.random.RandomState(self.seed)
            )
            best_hyper[case] = space_eval(space[case], case_best_pt)
            print(f'{case} training finisehd. Best param on this part is: ', best_hyper[case])
        
        # Now build the model based on best hyperparams 
        self.best_hyper = best_hyper
        self.model = self.meta_model(self.best_hyper, self.ensemble)
        for case in self.mystl_name:
            # if isinstance(self.model[case], Prophet):
            #     with suppress_stdout_stderr():
            #         self.model[case].fit(di['prophet_tra'])
            # else:
            self.model[case].fit(di[f'X_tra'], di[f'{case}_y_tra'])
        # check potential poor performance 
        tmp1 = self.predict(di, at=1, on='tra', aggregate=False) 
        tmp2 = self.predict(di, at=self.horizon, on='tra', aggregate=False) 
        for i, case in enumerate(self.mystl_name):
            if (tmp1[i] == tmp2[i]).all(): 
                warnings.warn(
                    f"In {case} same prediction between first and last horizon. "+\
                    "Performance may suffer."
                )
        # comment this if debug mode 
        del trials
        
        return self 
    
    def _validate_model(self): 
        """validate the model call"""
        if not hasattr(self, 'model'):
            raise AttributeError("attribute `model` not found. please use fit(...,on='tra') firstly.")
    
    def _validate_convert_at(self, at): 
        """validate the input value of `at` and explain at==-1 into the correct horizon"""
        assert at in range(1, self.horizon+1) or at in [-1], 'Invalid value for the argument `at`'
        if at == -1 : 
            at = self.horizon
        return at 
    
    def get_gt_predcit_data(self, di, at=-1):
        """
        Generate dataframe that contains both ground truth and prediction with datetimes aligned. 
        TODO: not yet include option for adding the out-of-data prediction based on di['X_final'].
        Args:
            di (dict): the dictionary generated from process_data() function
            at (int, optional): Determine which forecast horzion to predict. -1 means to predict on the last horizon.  
            all means predict on all horizons.. Defaults to -1.

        Returns:
            pd.DataFrame: the output that concats ground truth and predictions with aligned datetimes.
        """
        # initial check 
        self._validate_model()
        at = self._validate_convert_at(at)
        target = di['STL'].columns.str.replace('|'.join('_'+self.mystl_name), '', regex=True).unique()
        freq = di['info']['target_freq']
        # get prediction and gt 
        tra_mask = (di["X_tra"].dummy == at)
        tes_mask = (di["X_tes"].dummy == at)
        tra_pred = self.predict(di, at=at, on='tra', aggregate=True)
        tes_pred = self.predict(di, at=at, on='tes', aggregate=True)
        # pretty format 
        tra_pred = pd.DataFrame(tra_pred, columns=target+'_pred', index=di["X_tra"][tra_mask].asfreq(freq).index.shift(at))
        tes_pred = pd.DataFrame(tes_pred, columns=target+'_pred', index=di["X_tes"][tes_mask].asfreq(freq).index.shift(at))
        # merge gt with pred and return 
        output = di["STL"][target].merge(
            tra_pred.append(tes_pred), 
            left_index=True, right_index=True, how='outer'
        )
        return output

    def _prophet_predict(self, case: str, di, at, on, return_ci=False): 
        """prophet version predict function"""
        m = self.model[case]
        if on == 'tra': 
            pred = m.predict(di['prophet_tra'].append(di['prophet_tes'].iloc[:at, :])).iloc[at:]
        elif on == 'tes': 
            end = -self.horizon+at
            if end == 0: 
                end = None 
            pred = m.predict(di['prophet_tes']).iloc[at:end]
        elif on == 'final': 
            future = di['prophet_tes'][['ds']].shift(self.horizon).iloc[-self.horizon:]
            pred = m.predict(future)
        
        if return_ci: 
            ci = pred["yhat_lower", "yhat_upper"]
            return pred.yhat, ci
        return pred.yhat.values
    
    def predict(self, di, at=-1, on='tes', aggregate=True):
        """
        Generate prediction from STLModel model

        :di: dict | array-like - 
            If a dict, it stores all tra and tes datasets of trend/season/armma
            If an np.array() or pd.DataFrame/Series, make ``at`` th-forecast on it 
            unless ``on`` =='final' (which will then make all forecasts.
        :at: forced to be [1,2,...,horizon] when on='final'.
            Determine which forecast horzion to predict. -1 means to predict on the last horizon.  
            all means predict on all horizons.
        :on: ['tra', 'tes', 'final'] - generate prediction on training or testing dataset or 
            final out-of-data prediction
        :aggregate: bool - add up the prediction of three parts before return if true
        """

        self._validate_model()
        at = self._validate_convert_at(at)
        assert on in ['tra', 'tes', 'final'], 'Invalid value for the argument `on`.'
        
        # case normal prediction 
        # get prediciton of each part 
        X_on = di[f'X_{on}'] if isinstance(di, dict) else di 
        pred_matrix = []
        for case in pd.Index(self.model.keys()).drop("ensemble", errors="ignore"):
            # predict based on model type 
            # if isinstance(self.model[case], Prophet):
            #     pred = self._prophet_predict(case, di, at, on, return_ci=False)
            # else:
            # decide the pointer, ie, which data to use
            pointer =  X_on if on=='final' else X_on[X_on.dummy==at]
            pred = self.model[case].predict(pointer).reshape(-1, len(di['info']['targets']))
            pred_matrix.append(pred)
        pred_matrix = np.array(pred_matrix)     # 3-dim: 3 x sample_size x #target
        # return 
        if aggregate: 
            return pred_matrix.sum(axis=0)
        else: 
            return pred_matrix
    
    def get_feature_importance(self, di, mode='shap', **kwargs):
        """
        Get the feature importance either via tree built-in attributes or via shap algorithm based on ``mode``.
        Args:
            di (dict): the dictionary generated from process_data() function
            mode (str, optional): determine whild method to get the importance. The value can only be in ['tree', 'shape']. Defaults to 'shap'.

        Returns:
            return the list of feature importances for each of the target feature if mode is tree. 
            return shap values if mode is shap.
        """
        # check 
        assert mode in ['shap', 'tree'], "argument mode value is not valid."
        # init 
        
        if mode == 'tree':
            return self._tree_feature_importance(di, **kwargs)
        else: 
            return self._shap_feature_importance(di, **kwargs)
    
    def _shap_feature_importance(self, di, **kwargs): 
        # optional kwargs can have: `at`: int, `plot`: bool

        # initial check 
        self._validate_model()
        at = kwargs.get('at', -1)
        at = self._validate_convert_at(at)
        plot_shap = kwargs.get('plot', True)
        assert isinstance(plot_shap, bool), "kwargs argument `plot_shap` is invalid."
        # helper function. input: (#sample, #featuers) output: (#samples, # model outputs)
        def predict_for_shap(X):
            pred_matrix = []
            for case in pd.Index(self.model.keys()).drop("ensemble", errors="ignore"):
                pred = self.model[case].predict(X)
                if pred.ndim == 1: 
                    pred = pd.reshape(-1, 1)
                pred_matrix.append(pred)
            pred_matrix = np.array(pred_matrix) # 3-dim: 3 x sample_size x #targets
            return pred_matrix.sum(axis=0)

        # initializations
        targets = di['info']['targets']
        tra_mask = (di["X_tra"].dummy == at)
        tes_mask = (di["X_tes"].dummy == at)
        # rather than use the whole training set to estimate expected values, we could summarize with
        # a set of weighted kmeans, each weighted by the number of points they represent.
        X_tra_summary = shap.kmeans(di["X_tra"][tra_mask], 12)
        shapobj = shap.KernelExplainer(predict_for_shap, X_tra_summary)
        # TODO: train the model only with numpy to auto solve warnings.
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("ignore")
            shap_values = shapobj.shap_values(di["X_tes"][tes_mask])
        # plot 
        if plot_shap:
            for i in range(len(targets)):
                shap.summary_plot(shap_values[i], di["X_tes"][tes_mask], plot_type='bar', title=f'Feature Importance Plot for {targets[i]}', class_names=targets[i])
            shap.summary_plot(shap_values, di["X_tes"][tes_mask], plot_type='bar', title='Feature Importance Plot', class_names=targets)
        return shap_values

    def _tree_feature_importance(self, di):
        # init
        origin = di['STL']
        target = origin.columns.str.replace('|'.join('_'+self.mystl_name), '', regex=True).unique()
        # prepare output
        fi_collector = [] 
        for case in self.mystl_name: 
            # get support 
            support = self.model[case]['sel'].get_support()
            support_cols = di["X_tra"].columns[support]
            # get fi 
            if isinstance(self.model[case]['est'], MultiOutputRegressor): 
                tmp = []
                for i in range(len(target)): 
                    tmp.append(self.model[case]['est'].estimators_[i].feature_importances_)
                fi = np.array(tmp).mean(axis=0)
            else: 
                fi = self.model[case]['est'].feature_importances_
            # put into df and collect
            sel_fi = pd.DataFrame(fi, index=support_cols, columns=[case])
            fi_collector.append(sel_fi)
        return fi_collector
    
    def plot_predict(self, di, at, freq=None, ax=None, plot_aggregate=True, plot_ci=True, 
                     sig_lvl=0.05, figsize=(25,15)): 
        """
        Plot the prediction of the at-th horizon and the ground truth for each part of the model. 

        :freq: str - target feature datetime frequency. If None, then the system will try to auto detect it. 
        :ax: Axes object - the matplotlib.pyplot.Axes object that stores data information of a figure. 
        :plot_aggregate: bool - If True, plot the aggregation prediction in the 4-th subplot besides the trend/season/arma prediction subplots.
        :plot_ci: bool - NotImplemented yet. TODO.
        :sig_lvl: float - decides the confidence interval band, NotImplemented yet. TODO. 
        :figsize: 2-tuple - the width and height of the whole plot.
        """

        # initial set up 
        ## checking
        self._validate_model()
        at = self._validate_convert_at(at)
        ## get tra_finaldate, target, original data, datecol_tra/tes, and freq
        tra_finaldate = di['X_tra'].index[-1]
        origin = di['STL']
        target = origin.columns.str.replace('|'.join('_'+self.mystl_name), '', regex=True).unique()
        datecol = origin.index[:-self.horizon]
        if freq is None:
            freq = datecol.inferred_freq
        datecol.freq = freq
        tra_date = datecol[datecol <= tra_finaldate].shift(at)
        tes_date = datecol[datecol > tra_finaldate].shift(at)
        outofdata_date = origin.index[-self.horizon:].shift(self.horizon)
        ## get prediction at the `at`
        trapred = self.predict(di, at, on='tra', aggregate=False) # 3-dim: 3 x sample_size x #targets
        tespred = self.predict(di, at, on='tes', aggregate=False) # 3-dim: 3 x sample_size x #targets
        finalpred = self.predict(di, on='final', aggregate=False) # 3-dim: 3 x sample_size x #targets
        assert len(tra_date) == trapred.shape[1] and len(tes_date) == tespred.shape[1]
        assert len(outofdata_date) == finalpred.shape[1]
        
        # for loop over each target 
        for j, ta in enumerate(target):
            # determine the plot rows and ax
            nrows = 4 if plot_aggregate else 3 
            fig, ax = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
            # for loop over each trend/season/arma
            for i, case in enumerate(self.mystl_name):
                # plot prediction of train/test and the ground truth
                ax[i].plot(tra_date, trapred[i, :, j], 
                        c='b', label='train-predict')
                ax[i].plot(tes_date, tespred[i, :, j], 
                        c='orange', label='test-predict')
                ax[i].plot(origin[f"{ta}_{case}"], 
                        c='k', alpha=.5, label='ground truth')
                ax[i].axvline(tra_finaldate, c='r', label='train/test split point')
                ax[i].plot(outofdata_date, finalpred[i, :, j], 
                        c='r', linestyle='--', label='out-of-data predict')
                # formatting 
                ax[i].set(ylabel = case)
                ax[i].legend()

            if plot_aggregate: 
                i+=1
                ax[i].plot(tra_date, trapred.sum(axis=0)[:, j], 
                        c='b', label='train-predict')
                ax[i].plot(tes_date, tespred.sum(axis=0)[:, j], 
                        c='orange', label='test-predict')
                ax[i].plot(origin[ta], 
                        c='k', alpha=.5, label='ground truth')
                ax[i].axvline(tra_finaldate, c='r', label='train/test split point')
                ax[i].plot(outofdata_date, finalpred.sum(axis=0)[:, j], 
                        c='r', linestyle='--', label='out-of-data predict')
                # formatting 
                ax[i].set(ylabel = 'Aggregate')
                ax[i].legend()

            plt.suptitle(f"prediction of {ta} at {at}-th horizon", fontsize=18)
            plt.tight_layout()
            plt.show()
        return None 
    
    def score(self, di, at=-1, on='tes', aggregate=True, metrics=[], metrics_config={}): 
        """
        Calculate the performance score of either overall(aggregate=True) or each part of the prediction at 
        the ``at``-th horizon on the `on` dataset. 
        
        :di: dict - the output that generated by process_data() function.
        :at: str or int - which horizon are we taking to generate the prediction.
        :on: str in ['tra', 'tes'] - which dataset are we taking to evaluate the model performance.
        :metrics: list - metrics that will be calculated. If =[], use the default value from MetricsCls.__init__()
        :metrics_config: dict - specify the configuration of each metric that will be calculated.
        """
        # check 
        self._validate_model()
        at = self._validate_convert_at(at)
        assert on in ['tra', 'tes'], 'Invalid value for the argument `on`.'

        target = di["STL"].columns.str.replace('|'.join('_'+self.mystl_name), '', regex=True).unique()

        # get y_pred: 3-dim: 3 x sample_size x #targets
        y_pred = self.predict(di, at, on, aggregate)
        # get y_true: 3-dim: 3 x sample_size x #targets
        y_true = []
        mask = di[f'X_{on}'].dummy == at # mask
        for case in self.mystl_name: 
            y_true.append( di[f'{case}_y_{on}'][mask] )
        y_true = np.array(y_true)
        if aggregate: 
            y_true = y_true.sum(axis=0)
        
        # calculate metrics
        ## follow the default  ['MAE', 'RMSE', 'MAPE', 'DS'] if metrics == [] in the MetricsCls setting
        ## but here I change the STLModel.score default version of MAPE to be 'selfmade'
        if metrics_config=={}: 
            metrics_config={'MAPE__version': 'selfmade'}
        obj = MetricsCls(metrics=metrics, config = metrics_config)
        # prepare output
        res = {}
        for j, ta in enumerate(target): 
            if aggregate: 
                res[ta] = obj.score(y_true[:, j], y_pred[:, j])
            else: 
                subres = {}
                for i, case in enumerate(self.mystl_name): 
                    subres[case] = obj.score(y_true[i,:,j], y_pred[i,:,j])
                res[ta] = subres
        return res


