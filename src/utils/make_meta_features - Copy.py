'''
Short script to generate meta features, i.e. forecasts from one model that are used as input to another model.
This should be performed as a loop in which the model is refit to generate features for each timepoint using only prior
 data.
'''
import pandas as pd
from statsmodels.tsa import holtwinters
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from tqdm import tqdm

def create_meta_feature_frame(df, horizon, lead=24, rolling_periods=None, ci_alpha=None):
    if not df.index.freq:
        raise ValueError("Series must have freq attribute set for confidence interval prediction")
    freq = df.index.freq
    ci_lower_list = []
    ci_upper_list = []
    metalist = []
    meta_feature_name = "meta__ETS"
    for last_train_date in tqdm(df.index[lead:], desc="Making meta features"):
        # model = ARIMA(df.loc[:last_train_date], order=(2,1,1), seasonal_order=(0,0,0,0), trend="t")
        if rolling_periods is None:
            fit_data = df.loc[:last_train_date]
        elif isinstance(rolling_periods, int) and (rolling_periods > 0):
            fit_data = df.loc[last_train_date - rolling_periods * freq:last_train_date]
        else:
            raise ValueError("Argument rolling_periods must be either None or a positive integer!")
        model = ETSModel(
            fit_data, trend="add", error="mul",
            damped_trend=True, seasonal=None,
            seasonal_periods=12, initialization_method="estimated")
        # model = holtwinters.ExponentialSmoothing(df.loc[:last_train_date], trend="additive",
        #                                          damped_trend=True, seasonal=None,
        #                                          seasonal_periods=12, initialization_method="estimated")
        results = model.fit(disp=False)
        fh = results.forecast(horizon)
        f = results.predict(start=last_train_date + 1 * freq, end = last_train_date + horizon * freq)
        metalist.append(
            pd.DataFrame(data=f.to_numpy().reshape((1, horizon)), index=pd.Index([pd.Timestamp(last_train_date)]),
                         columns=[f"{meta_feature_name}_mean_t{t + 1}" for t in range(horizon)]))
        if ci_alpha is not None:
            cis = results.get_prediction(start=last_train_date + 1 * freq, end=last_train_date + horizon * freq)
            cis = cis.pred_int(alpha=ci_alpha)
            ci_lower = cis.iloc[:,0]
            ci_lower_list.append(pd.DataFrame(data=ci_lower.to_numpy().reshape((1,horizon)),
                                              index=pd.Index([pd.Timestamp(last_train_date)]),
                                              columns=[f"{meta_feature_name}_lower_t{t+1}" for t in range(horizon)]))
            ci_upper = cis.iloc[:,1]
            ci_upper_list.append(pd.DataFrame(data=ci_upper.to_numpy().reshape((1,horizon)),
                                              index=pd.Index([pd.Timestamp(last_train_date)]),
                                              columns=[f"{meta_feature_name}_upper_t{t+1}" for t in range(horizon)]))
    metafeats = pd.concat(metalist, axis=0)
    if ci_alpha is not None:
        ci_lower = pd.concat(ci_lower_list, axis=0)
        ci_upper = pd.concat(ci_upper_list, axis=0)
        metafeats = pd.concat((metafeats, ci_lower, ci_upper), axis=1)
    return metafeats

if __name__ == "__main__":
    # Dataframe containing the target
    target = "WPU101704"
    df = pd.read_csv(r"../data/input/steel_w_indicators_expanded.csv", index_col=0, parse_dates=True)
    df = df[target].dropna()
    df.index.freq = "MS"
    horizon = 3

    # meta_features = create_meta_feature_frame(df, horizon)
    meta_features = create_meta_feature_frame(df, horizon, ci_alpha=0.05)
    savepath = "meta_features_ETS.csv"
    meta_features.to_csv(savepath)