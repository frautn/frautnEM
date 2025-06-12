#     frautnEM is a set of library functions to be used in courses of electromagnetism.
#     Copyright (C) 2025  Edgardo Palazzo (epalazzo@fra.utn.edu.ar)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# 20240703
def plotEfVectorHilo(Ehilo, E, Lambda, Q, X, **params):
    """
    Muestra los vectores del campo en 2D calculados con dos métodos distintos.
    Función para comparar los resultados del campo de un segmento con el de un
    hilo infinito.

    Parameters
    ----------
    Ehilo : function
        Campo vectorial del hilo infinito (3 variables que devuelve 3 componentes).
    E : function
        Una función de un campo vectorial (3 variables que devuelve 3 componentes).
    Lambda: float
        Densidad lineal de carga del segmento, en C/m.
    Q : list
        Q = [
            [q1,x1,y1,z1],
            [q2,x2,y2,z2],
            ...
            [qN,xN,yN,zN]
        ]
    X : tuple
        Posiciones donde se calcula el campo.
    limites : tuple
        Lmites de los ejes: [xmin, xmax, ymin, ymax]
    scale : float
        Regula la longitud de las flechas.

    *Además de los parámetros de matplotlib y quiver, por ejemplo:*
    length : float
    figsize : tuple
    title : string
    """

    figsize = params.get('figsize', (7,5))
    title = params.get('title', "Algunos vectores del campo eléctrico de un segmento.")
    scale = params.get('scale', 1)

    xmin, xmax, ymin, ymax = 0,0,0,0
    x_pos = []
    y_pos = []
    Ei = []
    Ej = []
    for x in X:
        Eii, Ejj, Ekk = E(x[0],x[1],x[2],Q)

        # Elige límites para cuando el parámetro límites no es informado.
        if x[0] > xmax:
            xmax = x[0]
        if x[0] < xmin:
            xmin = x[0]
        if x[1] > ymax:
            ymax = x[1]
        if x[1] < ymin:
            ymin = x[1]
        x_pos = np.concatenate((x_pos,x[0]), axis=None)
        y_pos = np.concatenate((y_pos,x[1]), axis=None)
        N = np.sqrt(Eii**2 + Ejj**2)*1.5
        Ei = np.concatenate((Ei, Eii/N), axis=None)
        Ej = np.concatenate((Ej, Ejj/N), axis=None)

    Eihilo = []
    Ejhilo = []
    for x in X:
        Eii, Ejj, Ekk = Ehilo(x[0],x[1],x[2],Lambda)
        N = np.sqrt(Eii**2 + Ejj**2)*1.5
        Eihilo = np.concatenate((Eihilo, Eii/N), axis=None)
        Ejhilo = np.concatenate((Ejhilo, Ejj/N), axis=None)


    # Creating plot
    fig, ax = plt.subplots(figsize = figsize)
    ax.quiver(x_pos, y_pos, Ei, Ej, angles='xy', scale_units='xy', scale=scale)
    ax.quiver(x_pos, y_pos, Eihilo, Ejhilo, angles='xy', scale_units='xy', scale=scale, color='blue')

    for q in Q:
        qq, xq, yq, zq = q
        # Elige límites para cuando el parámetro límites no es informado.
        if xq > xmax:
            xmax = xq
        if xq < xmin:
            xmin = xq
        if yq > ymax:
            ymax = yq
        if yq < ymin:
            ymin = yq

        if qq > 0:
            colorq = 'red'
        else :
            colorq = 'green'
        circ = plt.Circle((xq,yq), np.max(np.abs(X))*0.02, color=colorq)
        ax.add_patch(circ)
    # ax.set_title(title)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    # Se expanden los límites automáticos:
    xmax = xmax + (xmax - xmin)*0.2
    xmin = xmin - (xmax - xmin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    ymin = ymin - (ymax - ymin)*0.2

    limites = params.get('limites', [xmin,xmax,ymin,ymax])
    ax.axis(limites)
    ax.set_title(title)
    plt.show()
    # plt.close()