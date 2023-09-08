import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load historical stock price data (you can replace this with your dataset)
# For simplicity, we'll create a sample dataset here
data = {'Date': pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'),
        'Price': np.random.rand(366) * 100}

df = pd.DataFrame(data)

# Feature engineering (you can add more features as needed)
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

# Define the features (X) and target (y)
X = df[['Day', 'Month']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the results
plt.scatter(X_test['Day'], y_test, color='blue', label='Actual')
plt.plot(X_test['Day'], y_pred, color='red', label='Predicted')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()

