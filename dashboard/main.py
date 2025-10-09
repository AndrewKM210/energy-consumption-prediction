import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pycountry


API_URL = st.secrets.api_url
API_SECRET = st.secrets.api_secret


# Contains the ISO-2 values for all countries
countries_db = [x.alpha_2 for x in pycountry.countries]

# Set streamlit title
st.title("EU Energy Consumption Forecasts")

# Year selector
year = st.slider("Select Year", min_value=2000, max_value=2030, value=2025)

# Get prediction from FastAPI backend
headers = {"Authorization": f"Bearer {API_SECRET}"}
response = requests.get(f"{API_URL}", params={"year": year}, headers=headers)

# If reponse is okay, parse response and convert countries from ISO-2 to ISO-3 values
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    df = df[df["country"].isin(countries_db)]
    df["country"] = df["country"].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)
else:
    st.error("API request failed")
    df = pd.DataFrame(columns=["country", "TOE_HAB"])

# Plot choropleth map
if not df.empty:
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="ISO-3",
        color="TOE_HAB",
        hover_name="country",
        color_continuous_scale="Viridis",
        title=f"Predicted Energy Consumption in {year}",
        scope="europe",
    )
    fig.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # remove empty borders
    )

    st.plotly_chart(fig, use_container_width=True)
