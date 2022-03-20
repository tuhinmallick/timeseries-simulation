import pandas as pd

def rename_feature_imporance(df_features, df_dict):
    """ Function to rename the feature importance name based on hard coded feature names and cryptic names.

    Args:
        df_features (pandas.DataFrame): Feature importance data frame with column names "feature" and "target_variable_name".
        df_dict (pandas.DataFrame): Data dictionary with the full names and the short names from the raw data set.

    Returns:
        pandas.DataFrame: Updates the feature names in the "feature" column with the names in the hardcoded feature engineering list.
    """
    # Define feature engineering name list. TODO: This should be somehow integrated with regex to be less statitc.
    feature_engineering_list = [
        " (Season)", 
        " (Trend)", 
        " (ARMA)",
        " (Lag 1)",
        " (Lag 2)",
        " (Lag 3)",
        " (Lag 6)",
        " (Skew 3)", 
        " (Skew 6)", 
        " Crisis Periods",
        "",
        " (3 Month Avg)",
        " (6 Month Avg)",
        " (3 Month Std)", 
        " (6 Month Std)", 
        " (Lag 1, Skew 6)", 
        " (Lag 2, Skew 6)", 
        " (Lag 3, Skew 6)", 
        " (Lag 6, Skew 6)",
        " (Lag 3, Skew 3)",
        " (Lag 6, Skew 3)", 
        " (3 Month Avg, Lag 1)",
        " (3 Month Avg, Lag 2)",
        " (3 Month Avg, Lag 3)",
        " (3 Month Avg, Lag 6)",
        " (6 Month Avg, Lag 1)",
        " (6 Month Avg, Lag 2)",
        " (6 Month Avg, Lag 3)", 
        " (6 Month Avg, Lag 6)",
        " (3 Month Std, Lag 1)",
        " (3 Month Std, Lag 2)",
        " (3 Month Std, Lag 3)",
        " (3 Month Std, Lag 6)",
        " (6 Month Std, Lag 1)",
        " (6 Month Std, Lag 2)",
        " (6 Month Std, Lag 3)",
        " (6 Month Std, Lag 6)",
        " (Exponential Moving Avg 2)",
        " (Exponential Moving Avg 3)",
        " (Exponential Moving Avg 6)",
        " (Momentum 1-Month)",
        " (Momentum 2-Month)",
        " (Momentum 3-Month)",
        " (Momentum 6-Month)",
        ]

    cryptic_names_list = [
        '_season',
        '_trend',
        '_arma',
        '_lag1',
        '_lag2',
        '_lag3',
        '_lag6',
        '_skew3',
        '_skew6',
        'crisis_dummy',
        '',
        '_ma3',
        '_ma6',
        '_ms3',
        '_ms6',
        '_lag1_skew6',
        '_lag2_skew6',
        '_lag3_skew6',
        '_lag6_skew6',
        '_lag3_skew3',
        '_lag6_skew3',
        '_lag1_ma3',
        '_lag2_ma3',
        '_lag3_ma3',
        '_lag6_ma3',
        '_lag1_ma6',
        '_lag2_ma6',
        '_lag3_ma6',
        '_lag6_ma6',
        '_lag1_ms3',
        '_lag2_ms3',
        '_lag3_ms3',
        '_lag6_ms3',
        '_lag1_ms6',
        '_lag2_ms6',
        '_lag3_ms6',
        '_lag6_ms6',
        '_ema2',
        '_ema3',
        '_ema6',
        '_momentum1',
        '_momentum2',
        '_momentum3',
        '_momentum6',
        
    ]

    # Consolidate to dataframe.
    df_feng = pd.DataFrame(
        {
            "short_names": cryptic_names_list,
            "feature_engineering": feature_engineering_list
        }
    )
    
    # Rename the features first.
    for i, feature in enumerate(df_dict["Feature Name"].tolist()):
        df_features["feature"] = df_features["feature"].str.replace(feature, df_dict["Full Name"].tolist()[i], regex=True)

    # Rename the feature engineering part. 
    for i, feature in enumerate(df_feng["short_names"].tolist()):
        df_features["feature"] = df_features["feature"].str.replace(feature, df_feng['feature_engineering'].tolist()[i], regex=True)
        
    return df_features

def rename_correlation_dict_features(simulation_correlation_dict, df_dict):
    """ Function that renames the keys in the simulation correlation dict to prettified names.

    Args:
        simulation_correlation_dict (dict): Simulation dictionary where keys are to be renamed.
        df_dict (pandas.DataFrame): Dictionary with "Feature Names", cryptic names and the Full Names, prettified names.

    Returns:
        dict: Returns dict with renamed keys.
    """
    for i, feature in enumerate(df_dict["Full Name"].tolist()):
        if df_dict.loc[i, "Feature Name"] in simulation_correlation_dict:
            simulation_correlation_dict[feature] = simulation_correlation_dict.pop(df_dict.loc[i, "Feature Name"])
    
    return simulation_correlation_dict