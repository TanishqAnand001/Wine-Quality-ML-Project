import streamlit as st
import requests
import json

API_URL = "https://wine-quality-ml-project.vercel.app/predict"

st.set_page_config(layout="wide")
st.title("🍷 Red Wine Quality Predictor")
st.write("Input the chemical properties of a red wine to predict its quality.")

st.sidebar.header("Wine Features")


def get_user_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.4, step=0.1,
                                        format="%.1f")
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.1, max_value=1.6, value=0.7, step=0.01,
                                           format="%.2f")
        citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f")
        residual_sugar = st.number_input("Residual Sugar", min_value=0.9, max_value=16.0, value=1.9, step=0.1,
                                         format="%.1f")

    with col2:
        chlorides = st.number_input("Chlorides", min_value=0.01, max_value=0.62, value=0.076, step=0.001, format="%.3f")
        free_sulfur = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=72.0, value=11.0, step=1.0,
                                      format="%.0f")
        total_sulfur = st.number_input("Total Sulfur Dioxide", min_value=6.0, max_value=289.0, value=34.0, step=1.0,
                                       format="%.0f")
        density = st.number_input("Density", min_value=0.990, max_value=1.004, value=0.9978, step=0.0001, format="%.4f")

    with col3:
        ph = st.number_input("pH", min_value=2.7, max_value=4.0, value=3.51, step=0.01, format="%.2f")
        sulphates = st.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.56, step=0.01, format="%.2f")
        alcohol = st.number_input("Alcohol", min_value=8.4, max_value=15.0, value=9.4, step=0.1, format="%.1f")

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur,
        "total sulfur dioxide": total_sulfur,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    return data


input_data = get_user_inputs()

st.subheader("Prediction")
if st.button("Predict Wine Quality"):
    if not API_URL.startswith("https"):
        st.error("Please update the API_URL in the streamlit_app.py file with your Vercel deployment URL.")
    else:
        try:
            payload = json.dumps(input_data)
            headers = {"Content-Type": "application/json"}

            response = requests.post(API_URL, data=payload, headers=headers)

            if response.status_code == 200:
                prediction = response.json()['prediction']
                if prediction == "Good Quality":
                    st.success(f"Prediction: **{prediction}** 🎉")
                else:
                    st.error(f"Prediction: **{prediction}** 👎")
            else:
                st.error(f"Error from API (Status {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")