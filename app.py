import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Train Delay Prediction",
    layout="wide"
)

df = pd.read_csv("train_delay.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.sidebar.title("Train Delay Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dashboard",
        "Prediction",
        "ℹAbout"
    ]
)

if page == "Home":

    st.title("Real-Time Passenger Train Delay Prediction")

    st.write("""
Welcome to the Train Delay Prediction System.

This application predicts the expected delay of passenger trains
using Machine Learning.
    """)

    st.subheader("Project Information")

    st.write("- Dataset Size :", df.shape[0], "rows")
    st.write("- Number of Columns :", df.shape[1])
    st.write("- Machine Learning Models : 4")
    st.write("- Target :", "Actual_Delay")

    st.subheader("Technologies Used")

    st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
    """)

elif page == "Dashboard":

    st.title("Dataset Dashboard")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Graphs")

    st.image("delay_distribution.png")

    st.image("delay_trend.png")

    st.image("weather_distribution.png")

    st.image("weather_vs_delay.png")

    st.image("day_count.png")

elif page == "Prediction":

    st.title("Train Delay Prediction")

    train_id = st.selectbox(
        "Train ID",
        encoders["Train_ID"].classes_
    )

    route = st.selectbox(
        "Route",
        encoders["Route"].classes_
    )

    source = st.selectbox(
        "Source",
        encoders["Source"].classes_
    )

    destination = st.selectbox(
        "Destination",
        encoders["Destination"].classes_
    )

    day = st.selectbox(
        "Day of Week",
        encoders["Day_of_Week"].classes_
    )

    month = st.selectbox(
        "Month",
        encoders["Month"].classes_
    )

    weather = st.selectbox(
        "Weather",
        encoders["Weather"].classes_
    )

    holiday = st.selectbox(
        "Holiday",
        encoders["Holiday"].classes_
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0
    )

    previous_delay = st.number_input(
        "Previous Delay (Minutes)",
        min_value=0.0
    )

    average_speed = st.number_input(
        "Average Speed (km/h)",
        min_value=1.0
    )

    departure_hour = st.slider(
        "Departure Hour",
        0,
        23,
        8
    )

    departure_minute = st.slider(
        "Departure Minute",
        0,
        59,
        0
    )

    if st.button("Predict Delay"):

        input_data = pd.DataFrame([{
            "Train_ID": encoders["Train_ID"].transform([train_id])[0],
            "Route": encoders["Route"].transform([route])[0],
            "Source": encoders["Source"].transform([source])[0],
            "Destination": encoders["Destination"].transform([destination])[0],
            "Distance_km": distance,
            "Day_of_Week": encoders["Day_of_Week"].transform([day])[0],
            "Month": encoders["Month"].transform([month])[0],
            "Weather": encoders["Weather"].transform([weather])[0],
            "Temperature": temperature,
            "Holiday": encoders["Holiday"].transform([holiday])[0],
            "Previous_Delay": previous_delay,
            "Average_Speed": average_speed,
            "Departure_Hour": departure_hour,
            "Departure_Minute": departure_minute
        }])

        input_data = input_data[feature_columns]

        prediction = model.predict(input_data)

        delay = round(prediction[0], 2)

        st.success(f"Predicted Delay: {delay} Minutes")

        if delay <= 5:
            st.success("Train is expected to be On Time")

        elif delay <= 20:
            st.warning("Slight Delay Expected")

        else:
            st.error("Major Delay Expected")    

elif page == "ℹAbout":

    st.title("About Project")

    st.write("### Project Title")
    st.write("Real-Time Passenger Train Delay Prediction Using Machine Learning")

    st.write("### Problem Statement")
    st.write("""
This project predicts passenger train delays using Machine Learning
based on route, weather, previous delay, temperature,
distance and other operational factors.
""")

    st.write("### Algorithms Used")
    st.write("""
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
""")

    st.write("### Best Model")
    st.success("Gradient Boosting Regressor")

    st.write("### Developed Using")
    st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- Streamlit
""")
    
