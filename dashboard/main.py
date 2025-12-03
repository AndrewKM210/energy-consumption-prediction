import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pycountry
import joblib


# Secret API info
API_URL = st.secrets.api_url
API_SECRET = st.secrets.api_secret
COUNTRIES_DS = joblib.load("trained_models/countries.pkl")

# Contains the ISO-2 values for all countries
COUNTRIES_DB = [x.alpha_2 for x in pycountry.countries]


def on_change_selectbox():
    """Callback for when selecbox changes value."""
    # Get prediction from FastAPI backend
    headers = {"Authorization": f"Bearer {API_SECRET}"}
    response = requests.get(
        f"{API_URL}/predict-country", params={"country": st.session_state.selectbox}, headers=headers
    )

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
    st.session_state.preds_country = df


def on_change_slider():
    """Callback for when slider changes value."""

    # Get prediction from FastAPI backend
    headers = {"Authorization": f"Bearer {API_SECRET}"}
    response = requests.get(f"{API_URL}/predict-year", params={"year": st.session_state.slider}, headers=headers)

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
    st.session_state.preds_year = df


# Set streamlit title (with HTML to center title in wide mode)
# Optionally add 'color: #4CAF50;' after '...center;' for green color
st.markdown("<h1 style='text-align: center; '>EU Energy Consumption Forecasts</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center;'>First prediction may be slow due to cold booting the API</h3>",
    unsafe_allow_html=True,
)

# Detect the current Streamlit theme for figure colors
light_theme = st.context.theme.type == "light"
template_name = "plotly_white" if light_theme else "plotly_dark"
bg_color = "white" if light_theme else "#0e1117"

# Create two columns (with a wider map and a small gap)
col_map, _, col_line = st.columns([1.5, 0.2, 1])
st.set_page_config(layout="wide")

# Plot map
df = pd.DataFrame()
with col_map:
    # Year selector
    st.slider("Select Year", min_value=2025, max_value=2050, value=2025, on_change=on_change_slider, key="slider")

    # Obtain predictions from session state or initialize with prepared predictions
    if "preds_year" not in st.session_state:
        print("Reading year prediction for first time")
        st.session_state["preds_year"] = pd.read_csv("dashboard/prediction_2025.csv")
    df = st.session_state.preds_year

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
            labels={"country": "Country", "TOE_HAB": "TOE per capita"},
            template=template_name,
        )
        fig.update_layout(
            autosize=True,
            margin={"r": 0, "t": 30, "l": 0, "b": 0},  # remove empty borders
            paper_bgcolor=bg_color,
            geo=dict(bgcolor=bg_color, landcolor=bg_color),
        )

        st.plotly_chart(fig, use_container_width=True, key="map2")

# Plot country forecast
with col_line:
    # Country selector
    st.selectbox("Select country", COUNTRIES_DS, index=10, on_change=on_change_selectbox, key="selectbox")

    # Obtain predictions from session state or initialize with prepared predictions
    if "preds_country" not in st.session_state:
        print("Reading country prediction for first time")
        st.session_state["preds_country"] = pd.read_csv("dashboard/prediction_ES.csv")
    df = st.session_state.preds_country
    country = pycountry.countries.get(alpha_2=st.session_state.selectbox).name

    # Plot forecast for country
    if not df.empty:
        fig = px.line(
            df,
            x="year",
            y="TOE_HAB",
            markers=True,
            labels={"year": "Year", "TOE_HAB": "TOE per capita"},
            title="Predicted energy consumption in the next 10 years",
            subtitle=f"For {country}",
            template=template_name,
        )
        fig.update_layout(
            autosize=True, margin={"r": 0, "t": 40, "l": 0, "b": 0}, paper_bgcolor=bg_color, plot_bgcolor=bg_color
        )
        st.plotly_chart(fig, use_container_width=True, key="map1")

print("Dashboard updated")
