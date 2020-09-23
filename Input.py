# -*- coding: utf-8 -*-
import numpy as np


class Input:

    def __init__(self, x: np.array, shape: tuple, nombre: str = "Input_layer"):
        """EL
        shape: lista de shape debe ser ancho, alto y capas"""
        self.X = x
        self.shape = shape
        self.nombre = nombre
        self.capas = None
        self.index = 0
        assert x.shape == shape, f"El shape de los datos {x.shape[1:]} es diferente del input indicado {shape}"


    def __str__(self, ):
        return f"Input: {self.nombre}, shape= {self.shape}, type: {type(self.X)}"


if __name__ == '__main__':
    volumen_random = np.random.random(9)
    df = np.array([volumen_random/(1+np.square(x)) for x in range(10)])
    entrada = Input(x=df, shape=(9,))
    print(entrada.__dict__)
