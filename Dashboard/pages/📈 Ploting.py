import re
import sys
sys.path.append("../EDA")

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from dython import nominal

from eda import most_common_words

#st.write("Here's our first attempt at using data to create a table:")
#st.write(pd.read_csv('../data/processed/train.csv'))

data = pd.read_csv('../data/processed/train.csv')

def plot_hist(df, col):
    """
    This function takes a dataframe and column 
    and plots a histogram for the series.
    """
    
    fig = px.histogram(df[col], nbins=90, marginal='box', opacity=0.8, color_discrete_sequence=['indianred'], width=3000, height=750)
    return fig

def plot_horizontal_bar(df):
    """
    This function takes a dataframe and column 
    and plots a histogram for the series.
    """

    df = df[:10]

    my_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
    my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
    my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(df))] # <-- Quick gradient example along the Red/Green dimensions.

    fig = px.bar(df, x="Frequency", y="Word", title='Word Frequency Plot - Description', orientation='h', color='Frequency').update_yaxes(autorange='reversed')
    return fig

    #df.plot.barh(x='Word', y='Frequency', color=my_colors, title='Word Frequency Plot - Description', figsize=(20,10)).invert_yaxis()

def plot_corr_heatmap(df):
    """
    This function takes a dataframe and 
    plots a correlation matrix for the features.
    """

    class_dummies = pd.get_dummies(df['Class Index'])
    df = df.drop(columns=['Title', 'Description'])
    #data = pd.concat([data,class_dummies], axis=1)
    #fig, ax = plt.subplots()
    return nominal.associations(df, figsize=(20,15),mark_columns=True)

if __name__ == '__main__':
    most_common_words_df = most_common_words(data, 'Description')

    st.set_page_config(layout="wide")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            option = st.selectbox('Select column in the dataset to view its histogram:', list(data.columns))
            st.write('You selected:', option)
            st.plotly_chart(plot_hist(data, option), use_container_width=True)
        with col2:
            st.plotly_chart(plot_horizontal_bar(most_common_words_df), use_container_width=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.imshow(plot_corr_heatmap(data)['corr'], text_auto=True))
        