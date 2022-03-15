import pandas

from BASF_Metals.Models.src.smoothboost.main_regression_deploy import *
from BASF_Metals.Models.src.smoothboost.pipelib.reduce_memory_usage import reduce_mem_usage
from BASF_Metals.Models.src.smoothboost.pipelib.grange_and_correlate import grange_and_correlate, user_input_correlation_picker
from BASF_Metals.Models.src.smoothboost.pipelib.remove_missing_data import remove_features_with_na
from BASF_Metals.Models.src.smoothboost.pipelib import STLModel
from BASF_Metals.Models.src.smoothboost.pipelib.reduce_memory_usage import reduce_mem_usage


def feature_engineering_pipeline(df: pandas.DataFrame, target: str, horizon: int, pre_selected_features: list, forecast_type: str):
    # ===========================================================================================================
    #           CONFIGRUATION
    # ===========================================================================================================
    # df = reduce_mem_usage(df)
    df = df.dropna(subset=[target])
    target_series = df[[target]]

    indicator_cols = [c for c in df.columns if c not in target_series]
    indicator_cols = pre_selected_features
    
    if forecast_type == "proportional_diff":
        # ===========================================================================================================
        #           MULTIPLICATIVE FEATURES
        # ===========================================================================================================

        df = df.join(df[[target] + indicator_cols].ewm(alpha=0.6).mean().pct_change().add_suffix("_ewm_pct_change1"))
        df = df.join(df[[target] + indicator_cols].ewm(alpha=0.6).mean().pct_change(periods=2).add_suffix("_ewm_pct_change2"))

        ## Setting: add skew
        skew_setting = [6]  # should be at least 3
        df, _ = fe.add_skew(df, features=[target] + indicator_cols, window=skew_setting)
        
        # Trigonometric encoding of months
        df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
        
        # Drop the original raw indicators and target series to improve performance
        # df = df.drop(columns=[target] + indicator_cols)
    
    # NOTE the STLModel is taken from Wen-Kai's PipeGBM Model.
    elif forecast_type in ["absolute_diff" or "absolute"]:
        
        # ===========================================================================================================
        #           ADDITIVE FEATURES
        # ===========================================================================================================

        # variable with original column names
        column_names = indicator_cols
        ## inplacely add lags
        lag_setting = [1, 2, 3, 6]
        lagdf = fe.create_lag(df, features=column_names, lag=lag_setting, get_momentum=True, droplagna=False)
        df = pd.concat([df, lagdf], axis=1).dropna()
        lagdf_columns = [c for c in lagdf.columns]
        column_names = column_names+lagdf_columns
        ## add ma / ema
        ma_setting = [3, 6]
        df, _ = fe.add_MA(df, features=column_names, window=ma_setting)
        ema_setting = [.2, .4, .6, .8]
        df, _ = fe.add_EMA(df, features=column_names, alpha=ema_setting)

        ## setting: add std
        ms_setting = [3, 6]
        df, _ = fe.add_MS(df, features=column_names, window=ms_setting)
        ## setting: add skew
        skew_setting = [3, 6]  # should be at least 3
        df, _ = fe.add_skew(df, features=column_names, window=skew_setting)
        # setting:  also get decomposition on exog features (STL model will auto decompose the endog features)

        # df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
        rolling = 24
        stlhelper = STLModel.STLModel(horizon=horizon)
        df = stlhelper._make_STL(
            df, target=column_names,
            freq="MS", rolling_strategy='rolling', rolling=rolling, extract_strategy='nanmean'
        )
        # Trigonometric encoding of months
        df = fu.add_month_sin_cos(df, df.index.month.to_numpy(), "month", 12)
        # NOTE ACTIVATE for year and quarter
        df["year"] = df.index.year
        column_names.append('year')
        
        df["quater"] = df.index.year
        column_names.append('quater')
        
        # TODO add month sin cos for year and quarter. 


    # ===========================================================================================================
    #           CRYSIS DUMMY VARIABLES
    # ===========================================================================================================
    
    crisis_dummy = (df.index >= "2008-05-01") & ( df.index < "2008-12-01" )
    covid_dummy = (df.index >= "2020-02-01")  & ( df.index < "2020-12-01" )
    debt_ceiling_2011 = (df.index >= "2011-08-01") & ( df.index < "2012-01-01" )
    debt_ceiling_2013 = (df.index >= "2013-02-01") & ( df.index < "2013-07-01" )
    chip_shortage = (df.index >= "2021-05-01") & ( df.index < "2021-12-01" )
    df['crisis_dummy'] = crisis_dummy | covid_dummy | debt_ceiling_2011 | debt_ceiling_2013 | chip_shortage
    
    return df, target_series