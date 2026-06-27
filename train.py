import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv("train_delay.csv")

print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
df.info()
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
print(df["Weather"].value_counts())
print(df["Holiday"].value_counts())
print(df["Route"].value_counts())
print(df["Actual_Delay"].mean())
print(df["Actual_Delay"].max())
print(df["Actual_Delay"].min())

df = df.drop_duplicates()
df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())
df["Previous_Delay"] = df["Previous_Delay"].fillna(df["Previous_Delay"].median())
df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
print(df.isnull().sum())

plt.figure(figsize=(8,5))
plt.hist(df["Actual_Delay"], bins=20)
plt.title("Distribution of Train Delays")
plt.xlabel("Delay (Minutes)")
plt.ylabel("Frequency")
plt.savefig("delay_distribution.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df.index, df["Actual_Delay"], marker='o', markersize=2)
plt.title("Train Delay Trend")
plt.xlabel("Record Number")
plt.ylabel("Actual Delay")
plt.grid(True)
plt.savefig("delay_trend.png")
plt.show()

weather_count = df["Weather"].value_counts()
plt.figure(figsize=(8,5))
plt.bar(weather_count.index, weather_count.values)
plt.title("Weather Distribution")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.savefig("weather_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Weather", y="Actual_Delay", data=df)
plt.title("Weather vs Train Delay")
plt.savefig("weather_vs_delay.png")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="Day_of_Week", data=df)
plt.title("Day-wise Train Count")
plt.xticks(rotation=45)
plt.savefig("day_count.png")
plt.show()

fig = px.scatter(df,x="Distance_km",y="Actual_Delay",color="Weather",title="Distance vs Train Delay")
fig.write_html("distance_vs_delay.png")
fig.show()

fig = px.box(df,x="Weather",y="Actual_Delay",color="Weather",title="Weather vs Delay")
fig.write_html("weather_boxplot.png")
fig.show()

monthly = df.groupby("Month")["Actual_Delay"].mean().reset_index()
fig = px.bar(monthly,x="Month",y="Actual_Delay",color="Month",title="Average Monthly Delay")
fig.write_html("monthly_delay.png")
fig.show()

df["Scheduled_Departure"] = pd.to_datetime(df["Scheduled_Departure"],format="%H:%M")
df["Departure_Hour"] = df["Scheduled_Departure"].dt.hour
df["Departure_Minute"] = df["Scheduled_Departure"].dt.minute
df.drop("Scheduled_Departure", axis=1, inplace=True)
print(df.head())

encoders = {}

categorical_columns = [
    "Train_ID",
    "Route",
    "Source",
    "Destination",
    "Day_of_Week",
    "Month",
    "Weather",
    "Holiday"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")

X = df.drop("Actual_Delay", axis=1)
joblib.dump(list(X.columns), "feature_columns.pkl")
y = df["Actual_Delay"]
print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
joblib.dump(scaler, "scaler.pkl")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2 Score :", r2_score(y_test, y_pred_lr))

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred_dt))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("R2 Score :", r2_score(y_test, y_pred_dt))

rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred_rf))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2 Score :", r2_score(y_test, y_pred_rf))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf)
plt.title("Actual vs Predicted Delay")
plt.xlabel("Actual Delay")
plt.ylabel("Predicted Delay")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
plt.show()

importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.show()

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred_gb))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_gb)))
print("R2 Score :", r2_score(y_test, y_pred_gb))

results = {
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "R2 Score": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_gb)
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_dt),
        mean_absolute_error(y_test, y_pred_rf),
        mean_absolute_error(y_test, y_pred_gb)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test, y_pred_gb))
    ]
}

comparison_df = pd.DataFrame(results)
print(comparison_df)

best_model = comparison_df.loc[comparison_df["R2 Score"].idxmax(),"Model"]
print("\nBest Model :", best_model)

plt.figure(figsize=(8,5))
plt.bar(comparison_df["Model"], comparison_df["R2 Score"])
plt.title("Model Comparison (R² Score)")
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.savefig("comparison.png")
plt.show()

joblib.dump(gb, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Gradient Boosting Model Saved Successfully!")
