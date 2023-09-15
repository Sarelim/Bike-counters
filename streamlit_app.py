import streamlit as st
import pandas as pd
import numpy as np

st.title('Bike counter app')

data = pd.read_parquet(Path("data") / "train.parquet")

# Add an option to show the raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    
