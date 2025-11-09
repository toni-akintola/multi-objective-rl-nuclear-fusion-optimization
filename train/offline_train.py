import pickle

with open("data/offline_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

from d3rlpy.algos import CQLConfig

# Create CQL agent
cql = CQLConfig().create()

# Train the agent
cql.fit(dataset, n_steps=100000)
