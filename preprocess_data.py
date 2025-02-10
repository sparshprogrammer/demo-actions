import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
import matplotlib.pyplot as plt

nifty_df = pd.read_csv('NIFTY_50.csv')

nifty_df = nifty_df.fillna(method='ffill')

nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], format='%Y-%m-%d')
df = nifty_df

# Feature engineering: Adding technical indicators to improve the model
df['Returns'] = df['Close'].pct_change()
df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
df['RSI'] = 100 - (100 / (1 + df['Returns'].rolling(window=14).mean()))  # Relative Strength Index
df = df.dropna()
df['Target'] = df['Close'].shift(+1)

train_data = df[(df['Date'] < '2020-02-28')]
test_data = df[(df['Date'] > '2020-02-28')]

model = ARIMA(train_data['Target'], exog=train_data[['Close']], order=(5, 1, 0))  # Including exogenous variables (indicators)
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.forecast(steps=len(test_data), exog=test_data[['Close']])

plt.figure(figsize=(10,6))
plt.plot(train_data['Close'], label="Train Data")
plt.plot(test_data['Close'], label="Test Data")
plt.plot(test_data.index, predictions, label="Predictions", color='yellow')
plt.title("Nifty 50 Predictions with SMAs and RSI")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
plt.savefig('./your_plot.png')

mae = mean_absolute_error(test_data['Close'], predictions)
print(f'Mean Absolute Error: {mae}')

# Use the Report to compare the reference data (train_data) with current data (test_data)
report = Report(metrics=[DataDriftPreset()])

# Create the drift report
report.run(reference_data=train_data[['SMA_20', 'SMA_50', 'RSI']], current_data=test_data[['SMA_20', 'SMA_50', 'RSI']], column_mapping=ColumnMapping())

# Step 5: Visualize the data drift
report.show()
report.save_html("./data_drift_report.html")
