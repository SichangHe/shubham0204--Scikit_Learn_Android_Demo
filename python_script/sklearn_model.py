import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression

# Reading the data from the CSV file
data = pd.read_csv("score.csv")

X, y = data.values[:, 0], data.values[:, 1]
X = np.expand_dims(X, axis=1)

# Fitting the linear regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Make predictions
print(f"Prediction for 8.5 hours is {regressor.predict([[8.5]])[0]}")


# Specify an initial type for the model ( similar to input shape for the model )
initial_type = [("input_study_hours", FloatTensorType([None, 1]))]

# Write the ONNX model to disk
converted_model = convert_sklearn(regressor, initial_types=initial_type)
with open("sklearn_model.onnx", "wb") as f:
    f.write(converted_model.SerializeToString())