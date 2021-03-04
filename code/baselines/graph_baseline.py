import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path

root = Path("../../data/")
root.mkdir(parents=True, exist_ok=True)

# read training data
df_train = pd.read_csv(
    root / 'train.csv', dtype={'authorID': np.int64, 'h_index': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv(root / 'test.csv', dtype={'authorID': np.int64})
n_test = df_test.shape[0]

# load the graph
G = nx.read_edgelist(root / 'collaboration_network.edgelist',
                     delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

# computes structural features for each node
core_number = nx.core_number(G)  # dict that associates node -> core_number
onion_number = nx.onion_layers(G)
avg_neighbor_degree = nx.average_neighbor_degree(G)
degree_centrality = nx.degree_centrality(G)
clustering = nx.clustering(G)

# create the training matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number and (3) the average degree of its neighbors
X_train_ = np.zeros((n_train, 6))
y_train_ = np.zeros(n_train)
for i, row in df_train.iterrows():
    node = row['authorID']
    X_train_[i, 0] = G.degree(node)
    X_train_[i, 1] = core_number[node]
    X_train_[i, 2] = avg_neighbor_degree[node]
    X_train_[i, 3] = onion_number[node]
    X_train_[i, 4] = degree_centrality[node]
    X_train_[i, 5] = clustering[node]
    y_train_[i] = row['h_index']

X_test_ = np.zeros((n_test, 6))
for i, row in df_test.iterrows():
    node = row['authorID']
    X_test_[i, 0] = G.degree(node)
    X_test_[i, 1] = core_number[node]
    X_test_[i, 2] = avg_neighbor_degree[node]
    X_test_[i, 3] = onion_number[node]
    X_test_[i, 4] = degree_centrality[node]
    X_test_[i, 5] = clustering[node]

scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train_)
X_test_ = scaler.fit_transform(X_test_)
# create the test matrix. each node is represented as a vector of 3 features:
# (1) its degree, (2) its core number and (3) the average degree of its neighbors
# X_test = np.zeros((n_test, 3))
# for i, row in df_test.iterrows():
#     node = row['authorID']
#     X_test[i, 0] = G.degree(node)
#     X_test[i, 1] = core_number[node]
#     X_test[i, 2] = avg_neighbor_degree[node]

X_train, X_test, y_train, y_test = train_test_split(
    X_train_, y_train_, test_size=0.2, random_state=42
)

# train a regression model and make predictions
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Loss: {mean_absolute_error(y_test, y_pred)}")

y_pred = model.predict(X_test_)
a
# write the predictions to file
output = Path('../../output')
output.mkdir(parents=True, exist_ok=True)

df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:, ["authorID", "h_index_pred"]].to_csv(
    output / 'test_predictions_graph.csv', index=False
)
