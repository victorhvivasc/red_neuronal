# -*- coding: utf-8 -*-
import numpy as np
from Input import Input
from comunes.funciones import inicializar_pesos, lineal, sigmoide, relu, derivada_sigmoide
np.random.seed(1)


class Densa:

    def __init__(self, neuronas, output=False, estimacion: str = 'lineal',
                 iniciar_bias=True, pesos='random', pesos_propios=None, nombre: str = 'Fully conected', capas=None,
                 index=0):
        """
        :param neuronas: int, numero de neuronas en una capa.
        :param output: int, se especifica unicamente en la ultima capa
        :param nombre: str, es un nombre a asignar a la capa
        :param estimacion: str: es el tipo de estimacion que se desea 'lineal', 'sigmoide'
        :param iniciar_bias: Booleano si es True inicia los pesos con un valor random
        :param pesos: str, indica el tipo de pesos que se desea inicializar ej 'zeros', 'random'
        :param pesos_propios: array de pesos predefinidos o obtenidos de entrenamientos previos
        """
        self.neuronas = neuronas
        self.output = output
        self.nombre = nombre
        self.iniciar_bias = iniciar_bias
        self.pesos_propios = pesos_propios
        self.pesos = pesos
        self.estimacion = estimacion
        self.z = np.zeros(self.neuronas)
        self.b = np.zeros(self.neuronas)
        self.pesos_propios = pesos_propios
        if isinstance(self.pesos, str):
            self.ws = np.array([])
        self._capas = capas
        self.parametros = 0
        self.index = index

    @property
    def capas(self, ):
        return self._capas

    @capas.setter
    def capas(self, capa_entrante):
        self._capas = capa_entrante

    def __call__(self, *args, **kwargs):
        #print(args[0].__dict__)
        if not args[0].capas:
            self.X = args[0].X
            if self.iniciar_bias:
                self.b = self.bias(self.neuronas)
            self.shape = self.X.shape[0]
            if isinstance(self.pesos, str):
                self.ws = inicializar_pesos(self, (self.shape, self.neuronas))
            self.capas = {args[0].index: args[0]}
            #self.z = self.forward()
            self.index += 1
            self.parametros = self.neuronas*len(self.ws)+len(self.b)
            self.capas[self.index] = self
        else:
            self.index = args[0].index + 1
            capa_previa = args[0].capas[args[0].index]
            self.X = np.copy(capa_previa.forward())
            if self.iniciar_bias:
                self.b = self.bias(self.neuronas)
            self.shape = self.X.shape[0]
            if isinstance(self.pesos, str):
                self.ws = inicializar_pesos(self, (self.shape, self.neuronas))
            self.z = self.forward()
            self.capas = args[0].capas.copy()
            self.parametros = self.neuronas*len(self.ws)+len(self.b)
            self.capas[self.index] = self

    @staticmethod
    def bias(outputs):
        return np.random.random(outputs)

    def forward(self,):
        for m in range(len(self.X)):
            for n in range(self.neuronas):
                #print(self.X.shape, self.ws.shape, self.b.shape)
                self.z[n] = lineal(self.X, self.ws[:, n], self.b[n])
                if self.estimacion == 'sigmoide':
                    self.z[n] = sigmoide(self.z[n])
                    self.b[n] = sigmoide(self.b[n])
                elif self.estimacion == 'relu':
                    self.z[n] = relu(self.z[n])
        return self.z

    def backpro(self, error, lr):
        for f in range(self.ws.shape[0]):
            for g in range(self.ws.shape[1]):
                self.ws[f][g] = self.ws[f][g] - lr*derivada_sigmoide(sigmoide(self.ws[f][g]))
                self.b[g] = self.b[g] + lr*derivada_sigmoide(sigmoide(self.b[g]))
                #print('bias', self.b)

if __name__ == '__main__':
    volumen_random = np.random.random(2)
    df = np.array([volumen_random/(1+np.square(x)) for x in range(4)])
    X = [[2, 4, 6], [7, 9, 11], [12, 14, 16], [17, 19, 21], [22, 24, 26], [27, 29, 31], [32, 34, 36], [37, 39, 41],
         [42, 44, 46], [47, 49, 51]]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    X = np.array([X])
    X = X/np.max(X)
    for n in range(len(X[0])):
        entrada = Input(x=X[0][n], shape=(3, ))
        densa1 = Densa(1, estimacion='sigmoide', pesos='random')
        densa1(entrada)
        print(densa1.forward())

