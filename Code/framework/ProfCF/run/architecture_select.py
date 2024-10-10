import pandas as pd


file_path = 'results/cf_grid_cf/agg/val.csv'


validation = pd.read_csv(file_path)


min_rmse_row = validation.loc[validation['rmse'].idxmin()]

# print(validation.columns)
columns = ['dataset', 'msg', 'gnn_layer', 'layers_num', 'stage', 'inter_func', 'act', 'cpnt_num', 'cpnt_aggr']


# print(min_rmse_row)

print(min_rmse_row[columns])