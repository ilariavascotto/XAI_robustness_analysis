import numpy as np
import scipy
import sklearn
import os
import argparse
import joblib
import torch

from numpy.linalg import norm
from scipy.stats import spearmanr as rho
from scipy.stats import kendalltau as tau

import utils.utils as ut
import load.load_dataset as ds
from load import load_net

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="pipeline")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--type", type=str, default="validation")
parser.add_argument("--agg", type=str, default="ensemble")
parser.add_argument("--neigh", type=str, default = "medoid")


args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")
elif args.model_name == "":
    raise ValueError("No model name was specified")
elif args.neigh not in ["medoid", "random"]:
    raise ValueError(f"You must specify a valid neighbourhood. There is no folder named results_{args.neigh}")

if args.agg in ["ensemble", "mean"]:
    agg = args.agg
else:
    raise ValueError("You must insert a valid aggregation method, either ensemble or mean.")

net = load_net.load_net(args.dataset)
dataset = ds.Dataset(args.dataset)

folder = os.path.join(os.getcwd(), f"results_{args.neigh}", f"{args.dataset}_{args.model_name}", args.type)

feature_names, categorical_features, categorical_names, ordinal, discrete, encoder, num_features = dataset.info()
X_train_or, X_train, y_train = dataset.train()
X_val_or, X_val, y_val = dataset.validation()
X_test_or, X_test, y_test = dataset.test()
model = dataset.load_model(args.model_name)

if args.type == "validation":
    data = X_val_or
    data_tensor = X_val
elif args.type == "test":
    data = X_test_or
    data_tensor = X_test
else:
    raise ValueError("Please insert a valid type: 'validation' or 'test'.")

num_points, num_ft_or = data.shape    
num_methods = 3

attr_path = os.path.join(folder, "attributions.npy")
neigh_size_path = os.path.join(folder, "neigh_size.npy")

attributions = np.load(attr_path)
neigh_size = np.load(neigh_size_path)

num = np.max(neigh_size)

print("Loaded")

if agg =="ensemble":
    aggregation = np.zeros(shape=(num_points, num, num_ft_or))
    for idx in range(num_points):
        dim_neigh = neigh_size[idx]
        for i in range(dim_neigh):
            aggregation[idx, i, :], _ = ut.ensemble(attributions[idx, i, : ,:])
    
    attributions = -np.abs(attributions)

elif agg == "mean":
    attr_norm = (attributions/norm(attributions, axis=2)[:, :, np.newaxis, :])
    attr = np.nan_to_num(attr_norm)

    mean = np.mean(attr, axis=3)
    mean = (mean/norm(mean, axis=2)[:, :, np.newaxis])

    aggregation = np.nan_to_num(mean)

np.save(os.path.join(folder, agg), aggregation)


robustness = np.zeros(shape = (num_points, num_methods +1))

for i in range(num_points):
    dim_neigh = neigh_size[i]

    if dim_neigh == 1:
        continue

    for j in range(num_methods):
        rho_ = rho(attributions[i, 0, :, j], attributions[i, 1:dim_neigh, :, j], axis=1).correlation
        robustness[i,j] = np.mean(rho_)
    
    rho_ = rho(aggregation[i,0,:], aggregation[i, 1:dim_neigh, :], axis=1).correlation
    robustness[i, 3] = np.mean(rho_)


np.save(os.path.join(folder, f"robustness_{agg}_{args.neigh}"), robustness)