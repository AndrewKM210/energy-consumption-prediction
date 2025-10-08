import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pycountry
import google.auth.transport.requests
from google.oauth2 import id_token


API_URL = st.secrets.api_url


@st.cache_data(ttl=3300)  # 
def get_gcp_token() -> str:
    """ 
        Use Cloud Run metadata identity to return identity token. 
        Use streamlit to cache token for 55 minutes, minimizes GCP requests.
    """
    auth_req = google.auth.transport.requests.Request()
    return id_token.fetch_id_token(auth_req, API_URL)


# Contains the ISO-2 values for all countries
countries_db = [x.alpha_2 for x in pycountry.countries]

# Set streamlit title
st.title("EU Energy Consumption Forecasts")

# Year selector
year = st.slider("Select Year", min_value=2000, max_value=2030, value=2025)

# Get GCP identity token
token = get_gcp_token()

# Get prediction from FastAPI backend
headers = {"Authorization": f"Bearer {token}"}
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
