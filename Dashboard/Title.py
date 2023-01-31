import streamlit as st
import pandas as pd

def main():
    """
    main function for app
    """

    pass

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.title("Article Classification App!")
        st.image("collection-newspapers.webp")

    with col3:
        st.write(' ')


