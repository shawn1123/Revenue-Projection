import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Read data from an Excel file
file_path = "your_excel_file.xlsx" 
df = pd.read_excel(file_path)

# "Date" column is in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Feature engineering: Extract additional features like year, quarter, etc.
df["Year"] = df["Date"].dt.year
df["Quarter"] = df["Date"].dt.quarter

# Split the data into training and testing sets
# train size can also be hardcoded accoring to your requirements
train_size = int(0.9 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Define the features and target variable
features = ["Year", "Quarter"]
target = "Revenue"

# Split the data into X and y
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Create and train an XGBoost regressor
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,  # Number of boosting rounds
    max_depth=3,       # Maximum tree depth
    learning_rate=0.1, # Learning rate
    random_state=42,
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model (MSE in this case)

# Create a Plotly figure for the actual and predicted revenue
fig = px.line(df, x="Date", y="Revenue", title="Quarterly Revenue Forecast")
fig.add_trace(px.line(test_data, x="Date", y=y_pred).data[0])
fig.show()
