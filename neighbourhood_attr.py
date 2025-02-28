import numpy as np
import scipy
import sklearn
import os
import argparse
import joblib
import torch


import captum
from utils.lrp import LRP
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

import gower
from scipy.stats import spearmanr as rho
from scipy.stats import kendalltau as tau

import utils.utils as ut
import utils.neighbourhood_generation as ng
import utils.utils_kmedoids as km
import load.load_dataset as ds
from load import load_net



import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="pipeline")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--type", type=str, default="validation")
parser.add_argument("--random", type= bool, default=False)
parser.add_argument("--alpha", type=float, default=0.05) # if random=True this is sigma
parser.add_argument("--alpha_cat", type=float, default=0.05) #if random=True this is gamma_cat
parser.add_argument("--k", type=int, default = 5)
parser.add_argument("--num", type=int, default=100)

args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)

if args.dataset == "":
    raise ValueError("No dataset was specified")
elif args.model_name == "":
    raise ValueError("No model name was specified")


net = load_net.load_net(args.dataset)
dataset = ds.Dataset(args.dataset)

parameters = joblib.load(os.path.join(os.getcwd(), "parameters.joblib"))
param = parameters[args.dataset]
n_clust = param["n_clust"]

if args.random:
    folder = os.path.join(os.getcwd(), "results_random", f"{args.dataset}_{args.model_name}")
else:
    folder = os.path.join(os.getcwd(), "results_medoid", f"{args.dataset}_{args.model_name}")

kmedoids_folder=os.path.join(os.getcwd(), "datasets", args.dataset, "kmedoids")

if not os.path.isdir(folder):
    os.makedirs(folder)
    print(f"New folder created at {folder}")

if not os.path.isdir(kmedoids_folder):
    os.makedirs(kmedoids_folder)
    print(f"New folder created at {kmedoids_folder}")

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

kmedoids_file = os.path.join(kmedoids_folder, "kmedoids.joblib")
centers_path = os.path.join(kmedoids_folder, "centers.npy")
knn_overall_path = os.path.join(kmedoids_folder, f"knn_overall_{args.k}.npy")
labels_path = os.path.join(kmedoids_folder, f"labels_{args.type}.npy")
labels_validation_path = os.path.join(kmedoids_folder, "labels_validation.npy")

bool_vars = [0 if i not in categorical_features else 1 for i in range(X_val_or.shape[1])]
check_cat = len(bool_vars) == sum(bool_vars)

if os.path.isfile(kmedoids_file):
    print("Recovering k-medoids of validation set")
    kmedoids = joblib.load(kmedoids_file)
    cluster_centers = np.load(centers_path)

    try:
        knn_overall = np.load(knn_overall_path) 
    except:
        if check_cat:
            knn_overall = km.knn_overall_manhattan(cluster_centers, n_neigh=args.k)
        else:
            knn_overall = km.knn_overall(cluster_centers, kmedoids.medoid_indices_, n_neigh=args.k, bool_vars=bool_vars)
        np.save(knn_overall_path, knn_overall)

    try:
        labels = np.load(labels_path)
    except:
        if check_cat:
            labels = km.km_predict_manhattan(X_test_or, kmedoids)
        else:
            labels = km.km_predict(X_test_or, cluster_centers, bool_vars)
        print(f"Saving k-medoids labels of {args.type} set to folder")
        np.save(labels_path, labels)

else:
    print("Computing k-medoids on validation set")

    if check_cat:
        kmedoids, cluster_centers, labels = km.km_manhattan(X_val_or, n_clust= n_clust)
        knn_overall = km.knn_overall_manhattan(cluster_centers, n_neigh=args.k)
    else:
        kmedoids, medoid_indices, cluster_centers, labels = km.km_gower(X_val_or, n_clust = n_clust, bool_vars=bool_vars )
        knn_overall = km.knn_overall(cluster_centers, medoid_indices, n_neigh=args.k, bool_vars=bool_vars)

    print("Saving k-medoids to folder")
    joblib.dump(kmedoids, kmedoids_file)
    np.save(centers_path, cluster_centers)
    np.save(labels_validation_path, labels)
    np.save(knn_overall_path, knn_overall)

    if args.type == "test":
        if check_cat:
            labels = km.km_predict_manhattan(X_test_or, kmedoids)
        else:
            labels = km.km_predict(X_test_or, cluster_centers, bool_vars)
        np.save(labels_path, labels)


num_points, num_ft_or = data.shape
num_methods = 3

if args.random:
    sigma = args.alpha
    gamma_cat = args.alpha_cat
else:
    alpha = args.alpha
    alpha_cat = args.alpha_cat

num = args.num

results = np.zeros(shape = (num_points, num+1, num_ft_or, num_methods))
neigh_size = []

for idx in range(num_points):
    cluster = int(labels[idx])
    target = int(np.argmax(model(data_tensor[idx, :]).detach().numpy()))

  
    if args.random:
        x = ng.random_neighbourhood(data[idx,:], num, sigma, gamma_cat, categorical_features, categorical_names, num_ft_or)
    else:
        x = ng.medoid_neighbourhood(data[idx,:], idx, labels, num, knn_overall, cluster_centers, alpha, alpha_cat, categorical_features, discrete, ordinal, categorical_names, num_ft_or)
        
    x = ng.keep_neighbourhood(x, target, model, encoder)



    dim_neigh = x.shape[0]
    neigh_size.append(dim_neigh)

    if dim_neigh == 1:
        print(f"No points in the neighbourhood for test point {idx}")
        continue

    attr = ut.compute_attributions(model, x, target, lrp_rule=GammaRule)

    for m_ in range(num_methods):
        results[idx, :dim_neigh, :, m_] = ut.reverse_encoding_neighbourhood(attr[:, :, m_], categorical_features, categorical_names, num_ft_or)

print("Saving results")
folder_type = os.path.join(folder, args.type)

if not os.path.isdir(folder_type):
    os.mkdir(folder_type)
    print(f"New folder created for {args.type}")


np.save(os.path.join(folder_type, "attributions"), results) #num_points x (num+1) x num_ft_or x 3
np.save(os.path.join(folder_type, "neigh_size"), neigh_size) #num_points
print("Saved attributions successfully.")

