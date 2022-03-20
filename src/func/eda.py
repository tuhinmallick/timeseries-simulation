import streamlit as st
import os,sys, logging, pathlib,pickle,traceback
src_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(src_location) not in sys.path:
    sys.path.append(os.path.realpath(src_location))
import utils.eda_base as eda_base

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#   EDA
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def exploratory_data_analysis(data):
    all_features = data.columns
    commodity = st.sidebar.selectbox(
        'Please select the target feature : ',
        all_features,  key = 'EDA_target_feature')
    target_name = commodity
    # target_name = f"{commodity}_spot_price"
    eda = None
    option = st.sidebar.selectbox(
        'Please select the type of plot : ',
        ['Time Series', 'Cross-Correlation', 'Monthly Box Plot','Yearly Box Plot', 'Granger Casaulity check'], key = 'EDA_type')
    st.sidebar.write('You have selected:', option)
    if data is not None:
        st.write('##### You have selected the commodity: ',commodity)
        eda = eda_base.exploratory_data_analysis(target_name = target_name, df=data)
        if option == 'Time Series':
            fig = eda.plotly_single_timeseries_plot(y_variable=target_name, rolling_mean=True, rolling_std=True, figsize=(1400, 500), streamlit=True, display_fig=False)
            st.plotly_chart(fig,  transparent=False )
        elif option == 'Cross-Correlation':
            features = data.drop(target_name, axis =1).columns
            feat = st.sidebar.selectbox(
        'Please select the feature you want to correlate : ',
        features, key = 'Cross_Correlation')
            fig = eda.plotly_single_correlation(y_variable=target_name, x_variable=feat,  max_lags=12, figsize=(1400, 500), streamlit=True, display_fig=False)
            st.plotly_chart(fig,  transparent=False )
        elif option == 'Monthly Box Plot':
            fig = eda.plotly_seasonal_boxplot_ym(y_variable=target_name,  box_group = "month", figsize=(1400, 500), streamlit=True, display_fig=False)
            st.plotly_chart(fig,  transparent=False )
        elif option == 'Yearly Box Plot':
            fig = eda.plotly_seasonal_boxplot_ym(y_variable=target_name,  box_group = "year", figsize=(1400, 500), streamlit=True, display_fig=False)
            st.plotly_chart(fig,  transparent=False )
        elif option == 'Granger Casaulity check':
            features = data.drop(target_name, axis =1).columns
            feat = st.sidebar.selectbox(
        'Please select the feature you want to check: ',
        features, key = 'Granger')
            fig = eda.plotly_single_granger(y_variable=target_name, x_variable=feat, max_lags=12, figsize=(1400, 500), streamlit=True, display_fig=False)
            st.plotly_chart(fig,  transparent=False )