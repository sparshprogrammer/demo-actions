import pandas as pd

nifty_df = pd.read_csv('NIFTY_50.csv')

nifty_df = nifty_df.fillna(method='ffill')
print(nifty_df)

from pycaret.time_series import *
nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], format='%Y-%m-%d')
# nifty_df.set_index('Date', inplace=True)
exp = setup(data=nifty_df, target='Close')
best_model = compare_models()
plot_model(best_model)

# Forecast the next 10 periods (for example, 10 days)
forecast = predict_model(best_model, fh=10)

# Show the forecasted results
print(forecast)
