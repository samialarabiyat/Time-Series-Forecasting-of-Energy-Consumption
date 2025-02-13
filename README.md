# Time-Series-Forecasting-of-Energy-Consumption
This project focuses on forecasting hourly energy consumption using machine learning techniques. The dataset used is PJME Hourly Energy Consumption, which is preprocessed and analyzed for feature extraction before training predictive models.


Key Components of the Code:
Data Download and Preprocessing:

The dataset is downloaded from a Kaggle source.
The script sets up a working environment by creating necessary directories.
The downloaded data is extracted and stored in a structured format.
Exploratory Data Analysis (EDA):

The dataset is loaded into a Pandas DataFrame.
The timestamp column is set as an index and converted to a datetime format.
Several visualizations are generated using Seaborn and Matplotlib to understand patterns in energy consumption.
Feature Engineering:

Time-based features such as hour, day of the week, quarter, month, and year are extracted from the timestamp.
The dataset is split into training and testing sets.
Machine Learning Model (XGBoost):

A XGBoost Regressor is trained to predict energy consumption.
The model is trained using time-based features, with hyperparameters tuned for optimal performance.
Feature importance is visualized to understand the impact of each feature on predictions.
Deep Learning Model (LSTM - Neural Network):

A Long Short-Term Memory (LSTM) network is implemented using TensorFlow/Keras.
The model is compiled using the Adam optimizer and trained on sequential data.
The LSTM model captures time dependencies in energy consumption patterns.
Forecasting and Evaluation:

The trained XGBoost model is used to make predictions on the test set.
A Root Mean Squared Error (RMSE) metric is calculated to evaluate the model’s accuracy.
The actual vs. predicted values are plotted to visually assess model performance.
Error Analysis:

The absolute error for each predicted day is calculated.
The worst and best predicted days are analyzed to understand model limitations.
