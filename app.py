import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.caption("Best of 5 algorithms compared with 10-fold cross-validation.")

@st.cache_resource
def load():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

saved = load()
model = saved["model"]
st.write(f"Model: **{saved['name']}**")

c1, c2 = st.columns(2)
with c1:
    inc = st.number_input("Median income (in $10,000s)", 0.5, 15.0, 3.8, 0.1)
    age = st.number_input("House age (years)", 1.0, 52.0, 29.0, 1.0)
    rooms = st.number_input("Avg rooms per household", 1.0, 15.0, 5.4, 0.1)
    bed = st.number_input("Avg bedrooms per household", 0.5, 5.0, 1.1, 0.1)
with c2:
    pop = st.number_input("Population of the block", 3.0, 35000.0, 1425.0, 50.0)
    occ = st.number_input("Avg household members", 1.0, 10.0, 3.0, 0.1)
    lat = st.number_input("Latitude", 32.5, 42.0, 34.2, 0.01)
    lon = st.number_input("Longitude", -124.4, -114.3, -118.5, 0.01)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([[inc, age, rooms, bed, pop, occ, lat, lon]], columns=saved["features"])
    price = float(model.predict(row)[0])
    st.success(f"Estimated median house value: **${price:,.0f}**")