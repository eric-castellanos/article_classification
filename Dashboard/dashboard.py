import streamlit as st
import pandas as pd
import plotly.express as px

#st.write("Here's our first attempt at using data to create a table:")
#st.write(pd.read_csv('../data/processed/train.csv'))

data = pd.read_csv('../data/processed/train.csv')

option = st.selectbox('Select column in the dataset to view its histogram:', list(data.columns))
st.write('You selected:', option)

def plot_hist(df, col):
    """
    This function takes a dataframe and column 
    and plots a histogram for the series.
    """
    
    fig = px.histogram(df[col], nbins=90, marginal='box', opacity=0.8, color_discrete_sequence=['indianred'], width=3000, height=750)
    return fig

# Plot!
st.plotly_chart(plot_hist(data, option), use_container_width=True)