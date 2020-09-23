# -*- coding: utf-8 -*-
import numpy as np # version 1.19


def inicializar_pesos(obj, like):
    if obj.pesos == 'zeros':
        return np.zeros(shape=like)
    elif obj.pesos == 'random':
        return np.random.random(like)


def lineal(x, w, b):
    lin = np.dot(x.T, w) + b
    return lin


def sigmoide(z):
    sig = 1/(1 + np.exp(-z))
    return sig


def relu(z):
    rel = max(z, 0)
    return rel


def derivada_sigmoide(sig):
    return sig*(1-sig)


def mse(pred, true):
    return np.square(pred - true)/2
