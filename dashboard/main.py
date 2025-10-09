import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pycountry


# Secret API info
API_URL = st.secrets.api_url
API_SECRET = st.secrets.api_secret

# Contains the ISO-2 values for all countries
COUNTRIES_DB = [x.alpha_2 for x in pycountry.countries]


def on_change_slider():
    """Callback for when slider changes value."""

    # Get prediction from FastAPI backend
    headers = {"Authorization": f"Bearer {API_SECRET}"}
    response = requests.get(f"{API_URL}", params={"year": st.session_state.slider}, headers=headers)

    # If reponse is okay, parse response and convert countries from ISO-2 to ISO-3 values
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df = df[df["country"].isin(COUNTRIES_DB)]
        df["country"] = df["country"].apply(lambda x: pycountry.countries.get(alpha_2=x).alpha_3)
    else:
        st.error("API request failed")
        df = pd.DataFrame(columns=["country", "TOE_HAB"])

    # Save predictions to session
    st.session_state.preds = df


# Set streamlit title
st.title("EU Energy Consumption Forecasts")

# Year selector
st.slider("Select Year", min_value=2025, max_value=2050, value=2025, on_change=on_change_slider, key="slider")

# Obtain predictions from session state or initialize with prepared predictions
if "preds" not in st.session_state:
    print("Reading for first time")
    st.session_state["preds"] = pd.read_csv("dashboard/prediction_2025.csv")
df = st.session_state.preds

# Plot map
if not df.empty:
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="ISO-3",
        color="TOE_HAB",
        hover_name="country",
        color_continuous_scale="Viridis",
        title=f"Predicted Energy Consumption in {st.session_state.slider}",
        scope="europe",
    )
    fig.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # remove empty borders
    )

    st.plotly_chart(fig, use_container_width=True)
