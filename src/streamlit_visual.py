# Import for excel downloading
import base64
import webbrowser
from datetime import timedelta
from io import BytesIO

# Dataframe standard packages
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching

# User settings
import settings

# Project packages
import wsb_reasoner

# Config streamlit page
st.set_page_config(
    page_title="Reddit Sentiment Analysis",
    page_icon="🚀",
    initial_sidebar_state="expanded",
    layout='wide'
)

# Sidebar
# st.sidebar.image(r'wsb_face.png', use_column_width='auto')


# Start page

# Get the last time (to check for updates in the db)
last_updated_utc = wsb_reasoner.get_last_upload_time()

@st.cache(max_entries=5)
def get_last_data_from_db(last_updated):
    
    # Get latest data
    comments_and_scores = wsb_reasoner.get_comments_and_scores_from_db()
    comments_and_scores = comments_and_scores.explode(column='tickers'). \
        reset_index(drop=True).rename(columns={'tickers': 'ticker'})

    return comments_and_scores


def get_top_frames(
        vader_compare,
        cut_off_now,
        cut_off_before,
        top=settings.top_elements):
    # Copy dataframe to change by value
    vader_compare = vader_compare.copy()

    # Cut-off times
    cut_off_now = pd.to_datetime(cut_off_now)
    cut_off_before = pd.to_datetime(cut_off_before)

    # Get dfs and merge in a single frame for comparison
    vader_1 = vader_compare[vader_compare.created_utc >= cut_off_now]. \
        groupby('ticker').agg({'ticker': 'count', 'compound': 'mean'}). \
        rename(columns={'ticker': 'mentions', 'compound': 'mean_sentiment'})
    print('Groupby 1')
    print(vader_1)
    vader_1 = vader_1.sort_values('mentions', ascending=False)
    
    print(vader_1)
    vader_1 = vader_1.add_suffix('_T1')
    print(vader_1)

    vader_0 = vader_compare[(vader_compare.created_utc >= cut_off_before) & (vader_compare.created_utc < cut_off_now)]. \
        groupby('ticker').agg({'ticker': 'count', 'compound': 'mean'}). \
        rename(columns={'ticker': 'mentions', 'compound': 'mean_sentiment'})
    print(vader_0)
    vader_0 = vader_0.sort_values('mentions', ascending=False)
    
    vader_0 = vader_0.add_suffix('_T0')

    vader_compare = vader_1.merge(vader_0, how='left', left_index=True, right_index=True)
    vader_compare.mean_sentiment_T0 = round(vader_compare.mean_sentiment_T0, 4)
    vader_compare.mean_sentiment_T1 = round(vader_compare.mean_sentiment_T1, 4)

    # Make comparisons
    vader_compare['delta_mentions'] = vader_compare.mentions_T1 / vader_compare.mentions_T0 - 1
    vader_compare['delta_score'] = vader_compare.mean_sentiment_T1 - vader_compare.mean_sentiment_T0

    # Prettify columns
    vader_compare = vader_compare[['mentions_T1', 'delta_mentions', 'mean_sentiment_T1', 'delta_score']]. \
        rename(columns={
        'mentions_T1': 'Mentions',
        'delta_mentions': '%Δ Mentions',
        'mean_sentiment_T1': 'Score',
        'delta_score': 'Δ Score'
    }).head(top)

    # Format deltas
    def red_green(cell):
        if type(cell) != str and cell < 0:
            return 'color: Crimson'
        elif type(cell) != str and cell >= 0:
            return 'color: ForestGreen'
        else:
            return 'color: black'

    # Format scores and mentions
    def red_blue_green(cell):
        if type(cell) != str and cell <= -0.05:
            return 'color: Crimson'
        elif type(cell) != str and cell >= 0.05:
            return 'color: ForestGreen'
        if type(cell) != str and -0.05 < cell < 0.05:
            return 'color: DarkCyan'
        else:
            return 'color: black'

    # Show stylized frame
    t_frame = vader_compare. \
        style.applymap(red_green, subset=pd.IndexSlice[:, ['%Δ Mentions', 'Δ Score']]). \
        format({'%Δ Mentions': '{:.2%}', 'Score': '{:.4}', 'Δ Score': '{:.4}'}, na_rep="New"). \
        apply(lambda row: np.repeat('background: lightgrey' if row.isnull().any() else '', row.shape[0]), axis=1). \
        applymap(red_blue_green, subset=pd.IndexSlice[:, ['Score']])

    return t_frame


# Get data
data = get_last_data_from_db(last_updated_utc)

# Start page
st.sidebar.title('Reddit  Sentiment Analysis')

# Get dates
min_date = (data.created_utc.min() + timedelta(hours=24)).to_pydatetime().date()
max_date = (data.created_utc.max() - timedelta(hours=24)).to_pydatetime().date()

# Ask for dates input
date = st.sidebar.date_input('Select day for top list preview:', min_value=min_date, max_value=max_date, value=max_date)

# Top list
with st.expander('Top ' + str(settings.top_elements) + ' mentions'):
  st.subheader('Top ' + str(settings.top_elements) + ' mentions for  ' + date.strftime("%A, %B %e, %Y"))
  st.table(get_top_frames(data, date, date - timedelta(days=1)))

explanation = """The score is the mean compound sentiment score for the horizon period.
Rule of thumb for interpretation: (>0.5) is positive; (<-0.5) is negative; else is neutral.

VADER (Valence Aware Dictionary and sEntiment Reasoner) analysis tool is applied whose scoring is accustomed to social network text and was tuned to recognize lexicon like smiles and emojis.
The top list shows top mentions and VADER scores for respective tickers.
  Mentions and scores are compared on daily basis.
  False positives are possible with one-letter tickers like $B and $F.
  The top list and insight tickers data are updated on an hourly basis.
  Tickers list for searching is from ALPHA VANTAGE.
"""
with st.expander('How to interpret the above table'):
  st.text(explanation)

st.title('Tickers insights')

# Add ticker selector to sidebar
ticker_to_show = st.sidebar.selectbox(
    label='Set ticker:',
    options=tuple(data.ticker.value_counts().keys().to_list())
)
st.sidebar.write(f'Number of tickers available : {data.shape[0]}')
# Add slider for time frames to use for aggregation
frequency = st.sidebar.slider(
    label='Set frequency for aggregation of scores and mentions:',
    min_value=0.25,
    max_value=6.0,
    value=0.75,
    step=0.25,
    format='%f hours',
    help= ' interval applies for both graphs below'
)
mentions_exp= st.expander('Mentions')
mentions_exp.subheader('Mentions by time for $' + ticker_to_show)

# Get charts data with scores reindex by frequency
chart_data_main = data[data.ticker == ticker_to_show].copy()
chart_data_main['Mentions'] = 1
chart_data_main = chart_data_main.groupby([pd.Grouper(
    freq=str(int(frequency * 60)) + 'min',
    key='created_utc'
)])[['neg', 'neu', 'pos', 'compound', 'Mentions']]. \
    agg({'neg': 'mean', 'neu': 'mean', 'pos': 'mean', 'compound': 'mean', 'Mentions': 'count'}). \
    rename(columns={'neg': 'Negative', 'neu': 'Neutral', 'pos': 'Positive', 'compound': 'Compound'})
chart_data_main.reindex(pd.date_range(
    chart_data_main.index.min(),
    chart_data_main.index.max(),
    freq=str(int(frequency * 60)) + 'min'
))
chart_data_main = chart_data_main.fillna(0)

# Mentions graph
chart_data = chart_data_main[['Mentions']]
mentions_exp.line_chart(chart_data)
sentiments_exp= st.expander('Sentiments')
sentiments_exp.subheader('Sentiments by time for $' + ticker_to_show)
sentiment_type = st.sidebar.selectbox(
    'Set sentiment:',
    ('Compound', 'Negative', 'Positive', 'Neutral')
)

# Split rows
# col1, col2 = st.columns([3, 8])

# Add slider for look-back period
look_back = st.sidebar.slider(
    label='Look-back period for moving averages:',
    min_value=6.0,
    max_value=48.0,
    value=8.0,
    step=0.25,
    format='%f hours'
)
span = int(look_back / frequency)
cols_for_chart = st.sidebar.multiselect(
    'Check which lines to plot on graph:',
    options=[
        sentiment_type,
        'EWM (' + str(look_back) + ' hours)',
        'EWM (' + str(look_back * 2) + ' hours)',
        'SMA (' + str(look_back) + ' hours)',
        'SMA (' + str(look_back * 2) + ' hours)'
    ],
    default=[
        'EWM (' + str(look_back) + ' hours)',
        'SMA (' + str(look_back) + ' hours)'
    ]
)

# Prepare data for chart
chart_data = chart_data_main[[sentiment_type]].copy()
chart_data['EWM (' + str(look_back) + ' hours)'] = chart_data[sentiment_type].ewm(span=span).mean()
chart_data['EWM (' + str(look_back * 2) + ' hours)'] = chart_data[sentiment_type].ewm(span=span * 2).mean()
chart_data['SMA (' + str(look_back) + ' hours)'] = chart_data[sentiment_type].rolling(span).mean()
chart_data['SMA (' + str(look_back * 2) + ' hours)'] = chart_data[sentiment_type].rolling(span * 2).mean()
chart_data = chart_data[cols_for_chart]

# Chart data
sentiments_exp.line_chart(chart_data)

# Comments downloader (removed for sake of RAM performance)
    # Place link to download comments for particular ticker
def df_to_excel_link(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data)

    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Click to ' \
           f'download comments and vader scores for ${ticker_to_show} in Excel format (Strong language!)</a>'


    # Download tickers comments with scores data
    ticker_comments = data[data.ticker == ticker_to_show].copy()
    ticker_comments = ticker_comments.sort_values(by='created_utc', ascending=False).reset_index(drop=True)
    st.markdown(df_to_excel_link(ticker_comments), unsafe_allow_html=True)
