from asyncio.windows_events import NULL
import os, sys, traceback
from numpy import empty
import streamlit as st
import pandas as pd
from PIL import Image

base_path = r"C:\Users\mallict\forecasty-lab"

if os.path.realpath(base_path) not in sys.path:
    sys.path.append(os.path.realpath(base_path))
import BASF_Metals.EDA.src.eda_base as eda_base
# from  BASF_Metals.Models.src.smoothboost.pipelib.yahoo_finance import yfinance as api 

# class webApp():

class stream():
    def __init__(self):
        self.bytes_data=None

    @st.cache(persist=True)
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.index = pd.DatetimeIndex(df.index, freq=df.index.inferred_freq)
        return df

    def upload():
        df =None
        try : 
            uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                df = stream.load_data(uploaded_file)
                st.sidebar.write("{} has been uploaded".format(uploaded_file.name))
        except Exception as err:
            st.sidebar.write("{} is not the proper file format".format(uploaded_file.name))
        return df

    def main():
        try : 
            # st.write(api.quote())
            image = Image.open(os.path.join(base_path, "BASF_Metals", "raw_data", "logo_forecasty.PNG"))
            st.sidebar.image(image, output_format ='PNG', use_column_width  ='always')
            st.title("PGM Price Forecasting")
            st.sidebar.write('Be the **God** and keep *forecasting* :sunglasses:')
            commodity = st.sidebar.radio(("Select your commodity among PGM"),
            ('Platinum', 'Palladium', 'Rhodium'))
            data = stream.upload()
            target_name = f"{commodity}_spot_price"
            eda = None
            option = st.sidebar.selectbox(
                'Please select the type of plot : ',
                ['Time Series', 'Cross-Correlation', 'Box Plot', 'Granger Casaulty check'])

            st.sidebar.write('You have selected:', option)
            if data is not None:
                st.write('##### You have selected the commodity: ',commodity)
                eda = eda_base.exploratory_data_analysis(target_name = target_name, df=data)
                if option == 'Time Series':
                    fig = eda.single_timeseries_plot(target_name, True, True, streamlit=True, transparent=False)
                    st.pyplot(fig,  transparent=False )
                elif option == 'Cross-Correlation':
                    features = data.drop(target_name, axis =1).columns
                    feat = st.sidebar.selectbox(
                'Please select the feature you want to correlate : ',
                features)
                    fig = eda.single_correlate_plot(y_variable=target_name, x_variable=feat, figsize=(20,14),file_name_addition=4, plot_transparent_backgorund=False, streamlit=True, fontsize_title=40, fontsize_label= 40 )
                    st.pyplot(fig,  transparent=False )
                elif option == 'Box Plot':
                    fig = eda.seasonal_boxplot_ym(y_variable=target_name,  streamlit=True, plot_transparent_backgorund=False)
                    st.pyplot(fig,  transparent=False )
                elif option == 'Granger Casaulty check':
                    features = data.drop(target_name, axis =1).columns
                    feat = st.sidebar.selectbox(
                'Please select the feature you want to check: ',
                features)
                    fig = eda.single_granger_plot(y_variable=target_name, x_variable=feat, max_lags=30, streamlit = True, save_path=None, figsize=(15,6), dpi=180,plot_transparent_backgorund=False,fontsize_title=20, fontsize_label= 20,fontsize_xyticks =20)
                    st.pyplot(fig,  transparent=False )

                # elif option == 'Box Plot':
                # elif option == 'Box Plot':

        except Exception as err:
            traceback.print_exc()
            print(err.args)






if __name__ == '__main__':
    stream.main()