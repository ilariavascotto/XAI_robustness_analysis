import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import captum
from captum.attr import IntegratedGradients, DeepLift
from utils.lrp import LRP
from utils.lrp import linear_rule, non_linear_support
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

import warnings
warnings.filterwarnings('ignore')

def compute_attributions(model, dataset, target, lrp_rule):
    ig = IntegratedGradients(model)
    dl = DeepLift(model)
    lrp = LRP(model)

    _ = linear_rule(lrp_rule)
    _ = non_linear_support(nn.Softmax)

    dataset.requires_grad_()

    METHODS = [ig, dl, lrp]

    attr = np.zeros(shape=(dataset.shape[0], dataset.shape[1], len(METHODS))) #num points x num features x num methods

    for i, method in enumerate(METHODS):
        tmp = method.attribute(dataset, target=target)
        tmp = tmp.detach().numpy()

        if i ==(len(METHODS)-1): #for lrp
            output_score = model(dataset).detach().numpy()[:, target]
            tmp = tmp* output_score[:,np.newaxis]

        attr[:, :, i] = tmp
    
    return attr

def reverse_encoding_neighbourhood(data, categorical_features, categorical_names, num_ft_or):
    try:
        data = data.detach().numpy()
    except:
        pass

    num_points, ft = data.shape

    lens = [len(categorical_names[i]) for i in categorical_names.keys()]
    acc = [int(np.sum(lens[:i])) for i in range(len(lens)+1)]

    numerical_features = [i for i in range(num_ft_or) if i not in categorical_features]
    
    X = np.zeros(shape = (num_points, num_ft_or))

    for idx, i in enumerate(range(len(acc)-1)):
        i_ = categorical_features[idx]
        X[:, i_] = np.sum(data[:, acc[i]:acc[i+1]], axis=1)

    for idx, j in enumerate(range(ft-acc[-1])):
        j_ = numerical_features[idx]
        X[:,j_] = data[:, j+acc[-1]]

    return X


def ensemble_weights(attr):
    return np.divide(np.std(attr), np.sqrt(np.abs(np.multiply(np.mean(attr), attr))))

def ensemble(reverse_attr, lambda_pen = 0.15):
    num_methods = reverse_attr.shape[1]
    
    weights = [ensemble_weights(attr) for attr in reverse_attr.T] 
    sign = [[1 if el >=0 else 0 for el in attr] for attr in reverse_attr.T]
    rank = np.argsort(np.argsort(-np.abs(reverse_attr), axis=0), axis=0)
    
    idx = np.where(np.std(reverse_attr, axis=0)==0)[0]    
    if len(idx)>0:
        for i in idx:
            rank[:,i] = np.full(fill_value = len(reverse_attr[:,i])-1, shape=reverse_attr[:,i].shape)
            weights[i] = np.full(fill_value = 1/1e-10, shape=reverse_attr[:,i].shape)
            sign[i] = [1 for i in range(len(reverse_attr[:,i]))]
    
    new_ensemble = np.zeros(reverse_attr.shape[0])
    sign=np.sum(sign, axis=0)
    penalization = [1 + lambda_pen*min(sign[i], num_methods-sign[i]) for i in range(sign.shape[0])]

    for i in range(num_methods):
        new_ensemble += np.multiply(rank[:,i], weights[i])
    
    new_ensemble = np.divide(new_ensemble, np.sum(weights, axis=0))
    new_ensemble = np.multiply(new_ensemble, penalization)

    sign = [0 if el<(num_methods+1)/2 else 1 for el in sign]
    return new_ensemble, sign
