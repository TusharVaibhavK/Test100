import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# Load and preprocess data


@st.cache_data
def load_and_train_model():
    df = pd.read_csv('train_delays.csv')
    X = df[['From', 'To', 'Time', 'Weather', 'Day Type']]
    y = df['Delay (min)']

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    feature_names = encoder.get_feature_names_out(
        ['From', 'To', 'Time', 'Weather', 'Day Type'])

    model = LinearRegression()
    model.fit(X_encoded, y)

    # Calculate RMSE for confidence interval
    y_pred = model.predict(X_encoded)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return df, encoder, model, feature_names, rmse

# Function to predict delay


def predict_delay(from_station, to_station, time, weather, day_type, encoder, model, feature_names):
    input_data = pd.DataFrame({
        'From': [from_station],
        'To': [to_station],
        'Time': [time],
        'Weather': [weather],
        'Day Type': [day_type]
    })
    input_encoded = encoder.transform(input_data)
    input_encoded_df = pd.DataFrame(input_encoded, columns=feature_names)
    delay = model.predict(input_encoded_df)[0]
    return delay


# Load data and model
df, encoder, model, feature_names, rmse = load_and_train_model()

# Add route column
df['Route'] = df['From'] + " to " + df['To']

# City coordinates for map (example data, expand as needed)
city_coords = {
    'Indore': [22.7196, 75.8577], 'Chennai': [13.0827, 80.2707], 'Mumbai': [19.0760, 72.8777],
    'Lucknow': [26.8467, 80.9462], 'Patna': [25.5941, 85.1376], 'Delhi': [28.7041, 77.1025],
    'Chandigarh': [30.7333, 76.7794], 'Nagpur': [21.1458, 79.0882], 'Bangalore': [12.9716, 77.5946],
    'Pune': [18.5204, 73.8567], 'Ahmedabad': [23.0225, 72.5714], 'Hyderabad': [17.3850, 78.4867],
    'Bhopal': [23.2599, 77.4126], 'Kolkata': [22.5726, 88.3639], 'Jaipur': [26.9124, 75.7873]
}
df['From_Lat'] = df['From'].map(lambda x: city_coords[x][0])
df['From_Lon'] = df['From'].map(lambda x: city_coords[x][1])
df['To_Lat'] = df['To'].map(lambda x: city_coords[x][0])
df['To_Lon'] = df['To'].map(lambda x: city_coords[x][1])

# Streamlit app
st.title("Train Delay Prediction App")

# Sidebar for navigation and theme
page = st.sidebar.selectbox(
    "Choose a page", ["Top 10 Trains", "Predict Delay"])
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], key="theme")
if theme == "Dark":
    st.markdown(
        "<style>body {background-color: #1E1E1E; color: white;}</style>", unsafe_allow_html=True)

# Page 1: Top 10 Trains
if page == "Top 10 Trains":
    st.header("Top 10 Trains with Highest Delays")

    # Top 10 table
    top_10 = df.nlargest(10, 'Delay (min)')[
        ['Route', 'Delay (min)', 'Time', 'Weather', 'Day Type']]
    st.write("### Top 10 Delayed Trains")
    st.dataframe(top_10)
    csv = top_10.to_csv(index=False)
    st.download_button("Download Top 10 as CSV", csv,
                       "top_10_delays.csv", "text/csv")

    # Bar chart
    st.write("### Bar Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_10['Route'], top_10['Delay (min)'], color='skyblue')
    ax.set_xlabel('Train Route')
    ax.set_ylabel('Delay (min)')
    ax.set_title('Top 10 Trains by Delay')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Weather impact
    st.subheader("Average Delay by Weather")
    weather_avg = df.groupby('Weather')['Delay (min)'].mean().sort_values()
    fig, ax = plt.subplots()
    ax.bar(weather_avg.index, weather_avg.values, color='lightgreen')
    ax.set_xlabel('Weather')
    ax.set_ylabel('Average Delay (min)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Delay distribution
    st.subheader("Delay Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Delay (min)'], bins=20, color='purple', alpha=0.7)
    ax.set_xlabel('Delay (min)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Heatmap
    st.subheader("Delay Heatmap by Route")
    pivot_table = df.pivot_table(
        values='Delay (min)', index='From', columns='To', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', ax=ax)
    st.pyplot(fig)

    # Route map
    st.subheader("Route Map")
    fig = px.scatter_geo(df, lat='From_Lat', lon='From_Lon',
                         size='Delay (min)', hover_name='Route')
    st.plotly_chart(fig)

    # Feature importance
    st.subheader("Feature Importance")
    coef_df = pd.DataFrame(
        {'Feature': feature_names, 'Coefficient': model.coef_})
    top_coef = coef_df.nlargest(10, 'Coefficient')
    fig, ax = plt.subplots()
    ax.bar(top_coef['Feature'], top_coef['Coefficient'], color='orange')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Coefficient')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Model metrics
    st.subheader("Model Performance")
    X_encoded = encoder.transform(
        df[['From', 'To', 'Time', 'Weather', 'Day Type']])
    y_pred = model.predict(X_encoded)
    st.write(f"R-squared: {r2_score(df['Delay (min)'], y_pred):.4f}")
    st.write(
        f"RMSE: {mean_squared_error(df['Delay (min)'], y_pred, squared=False):.2f} min")

# Page 2: Predict Delay
elif page == "Predict Delay":
    st.header("Predict Train Delay")

    with st.form(key='prediction_form'):
        from_station = st.selectbox(
            "From Station", sorted(df['From'].unique()))
        to_station = st.selectbox("To Station", sorted(df['To'].unique()))
        time = st.selectbox("Time of Day", sorted(df['Time'].unique()))
        weather = st.selectbox("Weather", sorted(df['Weather'].unique()))
        day_type = st.selectbox("Day Type", sorted(df['Day Type'].unique()))
        compare = st.checkbox("Compare Scenarios")
        submit_button = st.form_submit_button(label='Predict Delay')

    if submit_button:
        delay = predict_delay(from_station, to_station, time,
                              weather, day_type, encoder, model, feature_names)
        ci = 1.96 * rmse  # 95% confidence interval
        st.success(f"Predicted Delay: **{delay:.2f} minutes** (±{ci:.2f} min)")

        st.write(f"Route: {from_station} to {to_station}")
        st.write(f"Time: {time}")
        st.write(f"Weather: {weather}")
        st.write(f"Day Type: {day_type}")

        # Historical comparison
        similar_trips = df[(df['From'] == from_station) & (
            df['To'] == to_station) & (df['Weather'] == weather)]
        if not similar_trips.empty:
            st.write("Historical Delays for Similar Trips:")
            st.dataframe(similar_trips[['Time', 'Day Type', 'Delay (min)']])

        # Multi-scenario comparison
        if compare:
            weather_options = df['Weather'].unique()
            day_options = df['Day Type'].unique()
            results = []
            for w in weather_options:
                for d in day_options:
                    delay = predict_delay(
                        from_station, to_station, time, w, d, encoder, model, feature_names)
                    results.append(
                        {'Weather': w, 'Day Type': d, 'Delay (min)': delay})
            st.write("Scenario Comparison:")
            st.dataframe(pd.DataFrame(results))

# # Run instructions
# st.sidebar.write(
#     "To run this app, use the command: `streamlit run train_delay_app.py`")
