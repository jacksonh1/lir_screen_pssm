import pandas as pd


df = pd.read_csv('./SF3_input_postcollapse_JK.csv')
rn_dict = {i: i.replace('_x', '') for i in df.columns if '_x' in i}
df = df.rename(columns=rn_dict)
df.to_csv('./SF3_input_postcollapse_JK.csv', index=False)
