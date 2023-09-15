import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Bike counter app')
st.set_option('deprecation.showPyplotGlobalUse', False)

##EDA

train_data = pd.read_parquet("data/train.parquet")
test_data = pd.read_parquet("data/test.parquet")

# Add an option to show the raw data
if st.checkbox('Show raw train data'):
    st.subheader('Raw data')
    st.write(train_data)
    
# Display a map of the data
st.subheader('Map of all counters')
filtered_data = train_data[train_data["date"].dt.hour == 12]
st.map(filtered_data[["latitude","longitude","bike_count"]], zoom=13)



#plot some Preliminary data for EDA  
st.write("##### Bike Count Plot for Totem 73 boulevard de Sébastopol S-N")
plt.figure(figsize=(10, 5))

mask = train_data["counter_name"] == "Totem 73 boulevard de Sébastopol S-N"
train_data[mask].groupby(pd.Grouper(freq="1w", key="date"))[["bike_count"]].sum().plot()
st.pyplot()


##Linear regression

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

st.write ("### Linear Regression")

import problem



#Linear Regressor
from sklearn.linear_model import Ridge

X_train, y_train = _encode_dates(train_data.drop(columns=["bike_count", "counter_name", "site_name", "log_bike_count", "counter_installation_date", "counter_technical_id"])), train_data["log_bike_count"]
X_test, y_test = _encode_dates(test_data.drop(columns=["bike_count", "counter_name", "site_name", "log_bike_count", "counter_installation_date", "counter_technical_id"])), test_data["log_bike_count"]
X_train [["site_id", "counter_id"]] = X_train["counter_id"].str.split("-", expand=True)
X_test [["site_id", "counter_id"]] = X_test["counter_id"].str.split("-", expand=True)

regressor = Ridge()
regressor.fit(X_train, y_train)


#Visualization

st.write("##### Bike Count prediction for counter A")
mask = (
    (X_test["counter_id"] == "102007049")
    & (X_test["year"] == 2021)
    & (X_test["month"] == 8)
    & (X_test["day"] >= 20 )
)

df_viz = X_test.loc[mask].copy().dropna()
df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
df_viz["bike_count (predicted)"] = np.exp(regressor.predict(X_test[mask])) - 1

fig, ax = plt.subplots(figsize=(12, 4))

df_viz["bike_count"][200:].plot(ax=ax)
df_viz["bike_count (predicted)"][200:].plot(ax=ax, ls="--")
ax.set_title("Predictions with Ridge")
ax.set_ylabel("bike_count")
plt.legend()
st.pyplot()


st.write("##### Bike Count prediction for counter B")
mask = (
    (X_test["counter_id"] == "104057445")
    & (X_test["year"] == 2021)
    & (X_test["month"] == 8)
    & (X_test["day"] >= 20 )
)

df_viz = X_test.loc[mask].copy().dropna()
df_viz["bike_count"] = np.exp(y_test[mask.values]) - 1
df_viz["bike_count (predicted)"] = np.exp(regressor.predict(X_test[mask])) - 1

fig, ax = plt.subplots(figsize=(12, 4))

df_viz["bike_count"][200:].plot(ax=ax)
df_viz["bike_count (predicted)"][200:].plot(ax=ax, ls="--")
ax.set_title("Predictions with Ridge")
ax.set_ylabel("bike_count")
plt.legend()
st.pyplot()


st.write("As we can see, some counters are underestimated while some are overestimated: a basic linear regression fails to capture the complexity of the dataset.")