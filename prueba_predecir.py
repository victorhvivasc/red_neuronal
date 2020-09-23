from capa_densa import Densa
from Input import Input
import numpy as np
from comunes.funciones import mse

X = [[2, 4, 6], [7, 9, 11], [12, 14, 16], [17, 19, 21], [22, 24, 26], [27, 29, 31], [32, 34, 36], [37, 39, 41],
     [42, 44, 46], [47, 49, 51]]
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
X = np.array([X])
X = X/np.max(X)

errores = np.array([])
for e in range(1000):
    for n in range(len(X[0])):
        entrada = Input(x=X[0][n], shape=(3,))
        if e == 0:
            densa1 = Densa(5, estimacion='sigmoide', pesos='zeros')
            densa1(entrada)
            densa2 = Densa(1, estimacion='sigmoide', pesos='zeros')
            densa2(densa1)
        else:
            #print(densa1.ws, densa2.ws)
            densa1.pesos = None
            densa2.pesos = None
            #densa1 = Densa(2, estimacion='sigmoide', pesos_propios=densa1.ws[n])
            #densa1(entrada)
            #densa2 = Densa(1, estimacion='sigmoide', pesos_propios=densa2.ws[n])
            #densa2(densa1)
        error2 = mse(densa2.forward(), y[n])
        errores = np.append(errores, error2)
        densa2.backpro(errores.mean(), 0.00001)
        densa1.backpro(errores.mean(), 0.00001)
    if e % 80 == 0:
        print(f'error al ciclo {e}: ', error2)

y = np.array(y)
print(errores.mean())
print(densa2.ws)
print(densa1.ws)
#print(densa2.forward(), densa1.ws, densa2.ws)

