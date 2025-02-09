import pandas as pd

nifty_df = pd.read_csv('NIFTY_50.csv')

nifty_df = nifty_df.fillna(method='ffill')
print(nifty_df)

from pycaret.time_series import *
exp = setup(data=nifty_df, target='Close', time_index='Date')
best_model = compare_models()
print(best_model)
