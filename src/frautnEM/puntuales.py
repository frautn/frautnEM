#     frautnEM is a set of library functions to be used in courses of electromagnetism.
#     Copyright (C) 2024  Edgardo Palazzo (epalazzo@fra.utn.edu.ar)

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


# 20240815
def Ef(x, y, z, Q):
    """Calcula las componentes del campo eléctrico en N/C.
    Ingresar valores de x,y,z en metros y q en coulomb.
    Q es una lista de la forma:
    Q = [
        [q1,x1,y1,z1],
        [q2,x2,y2,z2],
        ...
        [qN,xN,yN,zN]
    ]
    """
    k = 9E9   #Constante de Coulomb en las unidades correspondientes.

    Ei,Ej,Ek = 0,0,0
    for qi in Q:
        r = ((x - qi[1])**2 + (y - qi[2])**2 + (z - qi[3])**2)**(3/2)
        Ei = Ei + k * qi[0] * (x - qi[1]) / r
        Ej = Ej + k * qi[0] * (y - qi[2]) / r
        Ek = Ek + k * qi[0] * (z - qi[3]) / r

    return Ei, Ej, Ek


# 20240719
def V(x,y,z,Q):
    """Calcula potencial eléctrico en Volt.
    Ingresar valores de x,y,z en metros y q en coulomb.
    Q es una lista de la forma:
    Q = [
        [q1,x1,y1,z1],
        [q2,x2,y2,z2],
        ...
        [qN,xN,yN,zN]
    ]
    """
    k = 9E9   #Constante de Coulomb en las unidades correspondientes.

    V = 0
    for qi in Q:
        r = ((x - qi[1])**2 + (y - qi[2])**2 + (z - qi[3])**2)**(1/2)
        V = V + k * qi[0] / r

    return V


# 20240717
# TODO: Return axs, add
# more control over plotting parameters.
# Add examples in the docstring.
def plotEf(Q, **params):
    """
    Muestra las líneas de campo eléctrico en 2D.

    Parameters
    ----------
    Q : list
        Q = [
            [q1,x1,y1,z1],
            [q2,x2,y2,z2],
            ...
            [qN,xN,yN,zN]
        ]
    dx : float
        Se produce una grilla con -dx <= x <= dx. Si dy = 0,
        se usan los mismos intervalos para esa variable: -dx <= y <= dx.
    dy : float (opcional)
        La grilla puede tener distintas dimensiones en cada eje.
    w : integer (opcional)
        Cantidad de particiones de cada dimensión en la grilla.

    *Además de los parámetros de matplotlib y streamplot, por ejemplo:*
    figsize : tuple
    title : string
    """

    dx = params.get('dx', 5)
    dy = params.get('dy', dx)
    w = params.get('w', 100)

    figsize = params.get('figsize', (5,5))
    title = params.get('title', 'Líneas de campo')
    linewidth = params.get('linewidth', 0.4)
    density = params.get('density', 0.7)

    # Convirtiendo w a número complejo se incluye el extremo del intervalo en mgrid.
    w = w * 1j
    Y, X = np.mgrid[-dx:dx:w, -dy:dy:w]
    Z = 0*X

    Ei, Ej, Ek = Ef(X,Y,Z,Q)

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    strm = axs.streamplot(X, Y, Ei, Ej, color='b',
                        linewidth=linewidth, density=density)
    for q in Q:
        qq, xq, yq, zq = q
        if qq > 0:
            colorq = 'red'
        else :
            colorq = 'green'
        circ = plt.Circle((xq,yq), dx*0.02, color=colorq)
        axs.add_patch(circ)
    axs.set_title(title)
    axs.set_xlabel('$x$ [m]')
    axs.set_ylabel('$y$ [m]')
    plt.grid()


#TODO: add 3d version.
def plotEfcontribuciones(Ef, Q, x, **params):
    """
    Muestra los vectores de cada porción de un cuerpo extenso, ¡en 2D!
    (no usar este código si las cargas están distribuidas en 3D).

    Parameters
    ----------
    Ef : function
        Una función de un campo vectorial (3 variables que devuelve 3 componentes).
    Q : list
        Q = [
            [q1,x1,y1,z1],
            [q2,x2,y2,z2],
            ...
            [qN,xN,yN,zN]
        ]
    X : tuple
        Posición donde se calcula el campo.
    limites : tuple
        Lmites de los ejes: [xmin, xmax, ymin, ymax]
    scale : float
        Regula la longitud de las flechas.
    r : float
        Radio de las partículas cargadas.
    linewidth : float
        Grosor de las líneas que muestran la dirección.
    in3D : bool
        If True, it produces a 3D graph.

    *Además de los parámetros de matplotlib y quiver, por ejemplo:*
    length : float
    figsize : tuple
    title : string
    """

    figsize = params.get('figsize', (5,5))
    title = params.get('title', "Contribuciones al campo eléctrico total")
    scale = params.get('scale', 1)
    linewidth = params.get('linewidth', 0.5)
    in3D = params.get('in3D', False)

    xmin, xmax, ymin, ymax, zmin, zmax = x[0],x[0],x[1],x[1], x[2], x[2]
    x_pos = []
    y_pos = []
    z_pos = []
    Ei = []
    Ej = []
    Ek = []
    modulos = []
    for q in Q:
        Eii, Ejj, Ekk = Ef(x[0],x[1],x[2],[q])
        # Elige límites para cuando el parámetro límites no es informado.
        if q[1] > xmax:
            xmax = q[1]
        if q[1] < xmin:
            xmin = q[1]
        if q[2] > ymax:
            ymax = q[2]
        if q[2] < ymin:
            ymin = q[2]
        if q[3] > zmax:
            zmax = q[3]
        if q[3] < zmin:
            zmin = q[3]
        x_pos = np.concatenate((x_pos,x[0]), axis=None)
        y_pos = np.concatenate((y_pos,x[1]), axis=None)
        z_pos = np.concatenate((z_pos,x[2]), axis=None)
        modulo = np.sqrt(Eii**2 + Ejj**2 + Ekk**2)
        modulos = np.concatenate((modulos, modulo), axis=None)
        Ei = np.concatenate((Ei, Eii), axis=None)
        Ej = np.concatenate((Ej, Ejj), axis=None)
        Ek = np.concatenate((Ek, Ekk), axis=None)

    # Se expanden los límites automáticos:
    xmax = xmax + (xmax - xmin)*0.2
    xmin = xmin - (xmax - xmin)*0.2
    ymax = ymax + (ymax - ymin)*0.2
    ymin = ymin - (ymax - ymin)*0.2
    zmax = zmax + (zmax - zmin)*0.2
    zmin = zmin - (zmax - zmin)*0.2

    rm = np.max(np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2))*0.01
    r = params.get('r', rm)

    Ei = np.round(Ei/np.max(modulos),3)
    Ej = np.round(Ej/np.max(modulos),3)
    Ek = np.round(Ek/np.max(modulos),3)

    if in3D:
        limites = params.get('limites', [xmin,xmax,ymin,ymax, zmin, zmax])
    else:
        limites = params.get('limites', [xmin,xmax,ymin,ymax])

    arrwidth = params.get('arrwidth', 0.005*(limites[1]-limites[0]))
    
    # Creating plot
    if in3D:
        pass
    else:
        fig, ax = plt.subplots(figsize = figsize)

        # ax.quiver(x_pos, y_pos, Ei, Ej, angles='xy', scale_units='xy', scale=scale)
        ax.quiver(x_pos, y_pos, Ei, Ej, scale=scale, width=arrwidth)

        # Necesito un for separado para determinar el tamaño de los círculos.
        for q in Q:
            qq, xq, yq, zq = q
            ax.plot([xq,x[0]], [yq,x[1]], color='b', linewidth=linewidth, linestyle='dashed')
            if qq > 0:
                colorq = 'red'
            else :
                colorq = 'green'
            circ = plt.Circle((xq,yq), r, color=colorq)
            ax.add_patch(circ)
        # ax.set_title(title)
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')

    ax.axis(limites)
    ax.set_title(title)
    plt.show()
    # plt.close()

# 20240819
def plotEfVector(Q, X, **params):
    """
    Muestra los vectores del campo en 2D usando pyplot.quiver.

    Parameters
    ----------
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

    figsize = params.get('figsize', (5,5))
    title = params.get('title', "Algunos vectores de campo eléctrico")
    scale = params.get('scale', 1)

    xmin, xmax, ymin, ymax = 0,0,0,0
    x_pos = []
    y_pos = []
    Ei = []
    Ej = []
    for x in X:
        Eii, Ejj, Ekk = Ef(x[0],x[1],x[2],Q)

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

    # Creating plot
    fig, ax = plt.subplots(figsize = figsize)
    ax.quiver(x_pos, y_pos, Ei, Ej, angles='xy', scale_units='xy', scale=scale)

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


# 20240819
def plotEfvector3d(Q, **params):
    """
    Muestra los vectores del campo de un sistema de cargas puntuales
    en 3D usando pyplot.quiver.

    Parameters
    ----------
    Q : list
        Q = [
            [q1,x1,y1,z1],
            [q2,x2,y2,z2],
            ...
            [qN,xN,yN,zN]
        ]
    dx,dy,dz : float
        Se produce una grilla con -dx <= x <= dx, -dy <= y <= dy, -dz <= z <= dz.
        Si solo se informa dx, se usa el mismo valor para dy y dz. dx=6 por defecto.
    w : integer (opcional)
        Cantidad de particiones de cada dimensión en la grilla.
    X,Y,Z: 1D, 2D or 3D array-like, optional
        The coordinates of the arrow locations. If dx is given, these are ignored.

    *Además de los parámetros de matplotlib y quiver, por ejemplo:*
    length : float
    figsize : tuple
    title : string
    """

    dx = params.get('dx', 6)
    dy = params.get('dy', dx)
    dz = params.get('dz', dx)
    w = params.get('w', 100)
    length = params.get('length', dx * 0.15)

    figsize = params.get('figsize', (4,4))
    title = params.get('title', 'Campo eléctrico')
    linewidth = params.get('linewidth', 0.4)

    # Convirtiendo w a número complejo se incluye el extremo del intervalo en mgrid.
    w = w * 1j
    X, Y, Z = np.mgrid[-dx:dx:w, -dy:dy:w, -dz:dz:w]

    Ei, Ej, Ek = Ef(X,Y,Z,Q)

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs = fig.add_subplot(projection='3d')
    axs.quiver(X, Y, Z, Ei, Ej, Ek, length=length, normalize=True)

    # Graficar las cargas.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xc = dx * 0.04 * np.outer(np.cos(u), np.sin(v))
    yc = dx * 0.04 * np.outer(np.sin(u), np.sin(v))
    zc = dx * 0.04 * np.outer(np.ones(np.size(u)), np.cos(v))

    for q in Q:
        qq, xq, yq, zq = q
        if qq > 0:
            colorq = 'red'
        else :
            colorq = 'green'
        axs.plot_surface(xc + xq, yc + yq, zc + zq, color=colorq)
    axs.set_title(title)
    axs.set_xlabel('$x$ [m]')
    axs.set_ylabel('$y$ [m]')
    plt.grid()

# Formatter para agregar V a las etiquetas de las equipotenciales.
def fmtV(x):
    return f"{x}V"

# 20240821
# Esta función puede mejorarse muchísimo, sobre todo respecto a las escalas y unidades.
def equipotencialesPuntuales(Q, dim = 1, niveles = 10, figsize=(6,6), titulo='Equipotenciales',
                EF = False, density=0.75, dq=0.02, **params):
    """
    Grafica equipotenciales generadas por la distribución de cargas Q.

    Parameters
    ----------
    Q : list
        Q = [
            [q1,x1,y1,z1],
            [q2,x2,y2,z2],
            ...
            [qN,xN,yN,zN]
        ]
    dim : integer (opcional)
        Valores máximos para x,y en cm.
    niveles : list
        Los valores de voltaje de las equipotenciales que se quiere graficar.

    *Además de los parámetros de matplotlib y quiver, por ejemplo:*
    length : float
    figsize : tuple
    title : string
    """

    if 'x' in params:
        x = params.get('x', 0)
        y = np.arange(-dim, dim+0.01, 0.01)
        z = np.arange(-dim, dim+0.01, 0.01)
        Y, Z = np.meshgrid(y, z)
        X = Y*0 + x
        Vmat = V(X,Y,Z,Q) 
        # Luego de calculados los potenciales,
        # reutilizo la grilla para las variables que se grafican.
        X, Y = np.meshgrid(y, z)
    elif 'y' in params:
        y = params.get('y', 0)
        x = np.arange(-dim, dim+0.01, 0.01)
        z = np.arange(-dim, dim+0.01, 0.01)
        X, Z = np.meshgrid(x, z)
        Y = X*0 + y
        Vmat = V(X,Y,Z,Q) 
        # Luego de calculados los potenciales,
        # reutilizo la grilla para las variables que se grafican.
        X, Y = np.meshgrid(x, z)
    else:
        z = params.get('z', 0)
        x = np.arange(-dim, dim+0.01, 0.01)
        y = np.arange(-dim, dim+0.01, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = X*0 + z
        Vmat = V(X,Y,Z,Q) 

    # Set the labels for the plane to be displayed.
    if isinstance(x, float) or isinstance(x, int):
        xlabel = 'y [m]'
        ylabel = 'z [m]'
    elif isinstance(y, float) or isinstance(y, int):
        xlabel = 'x [m]'
        ylabel = 'z [m]'
    elif isinstance(z, float) or isinstance(z, int):
        xlabel = 'x [m]'
        ylabel = 'y [m]'

    fig, ax = plt.subplots(1, 1, figsize=figsize,facecolor=(1, 1, 1) )
    ax.set_title(titulo)
    for carga in Q:
        q, xq, yq, zq = carga
        # Different colors for positive and negative charges.
        if q>0:
            color = 'red'
        else:
            color = 'blue'
        # Check if the charge has to be drawn or not.
        if isinstance(x, float) or isinstance(x, int):
            if xq == x:
                circ = plt.Circle((yq,zq), dq*dim, color=color)
                ax.add_patch(circ)
        elif isinstance(y, float) or isinstance(y, int):
            if yq == y:
                circ = plt.Circle((xq,zq), dq*dim, color=color)
                ax.add_patch(circ)
        elif isinstance(z, float) or isinstance(z, int):
            if zq == z:
                circ = plt.Circle((xq,yq), dq*dim, color=color)
                ax.add_patch(circ)

    if EF:
        CS2 = ax.contour(X, Y, Vmat, levels = niveles, colors = 'red', alpha=0.4)
        E = np.gradient(-1*Vmat)
        ax.streamplot(X, Y, E[1], E[0], linewidth=1, cmap=plt.cm.inferno,
              density=density, arrowstyle='->', arrowsize=1.5)
    else:
        CS2 = ax.contour(X, Y, Vmat, levels = niveles, colors = 'red', alpha=1)
    
    ax.clabel(CS2, inline=True, fmt=fmtV, fontsize=10)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

    # return Vmat

# # 20240719
# # Esta función puede mejorarse muchísimo, sobre todo respecto a las escalas y unidades.
# def equipotencialesPuntuales(Q, dim = 100, levels = 10, figsize=(6,6), titulo='Equipotenciales',
#                 EF = False, density=0.75, dq=0.02, **params):
#     """
#     Grafica equipotenciales generadas por la distribución de cargas Q.

#     Parameters
#     ----------
#     Q : list
#         Q = [
#             [q1,x1,y1,z1],
#             [q2,x2,y2,z2],
#             ...
#             [qN,xN,yN,zN]
#         ]
#     dim : integer (opcional)
#         Valores máximos para x,y en cm.
#     levels : list
#         Los valores de voltaje de las equipotenciales que se quiere graficar.

#     *Además de los parámetros de matplotlib y quiver, por ejemplo:*
#     length : float
#     figsize : tuple
#     title : string
#     """

#     if 'x' in params:
#         x = params.get('x', 0)
#         y = np.arange(-dim, dim+1)
#         z = np.arange(-dim, dim+1)
#         Y, Z = np.meshgrid(y, z)
#         X = Y*0 + x
#         Vmat = V(X,Y/100,Z/100,Q)  # Convertir Y, Z a metro.
#         # Luego de calculados los potenciales,
#         # reutilizo la grilla para las variables que se grafican.
#         X, Y = np.meshgrid(y, z)
#     elif 'y' in params:
#         y = params.get('y', 0)
#         x = np.arange(-dim, dim+1)
#         z = np.arange(-dim, dim+1)
#         X, Z = np.meshgrid(x, z)
#         Y = X*0 + y
#         Vmat = V(X/100,Y,Z/100,Q)  # Convertir X, Z a metro.
#         # Luego de calculados los potenciales,
#         # reutilizo la grilla para las variables que se grafican.
#         X, Y = np.meshgrid(x, z)
#     else:
#         z = params.get('z', 0)
#         x = np.arange(-dim, dim+1)
#         y = np.arange(-dim, dim+1)
#         X, Y = np.meshgrid(x, y)
#         Z = X*0 + z
#         Vmat = V(X/100,Y/100,Z,Q)  # Convertir X, Y a metro.

#     # Set the labels for the plane to be displayed.
#     if isinstance(x, float) or isinstance(x, int):
#         xlabel = 'y [cm]'
#         ylabel = 'z [cm]'
#     elif isinstance(y, float) or isinstance(y, int):
#         xlabel = 'x [cm]'
#         ylabel = 'z [cm]'
#     elif isinstance(z, float) or isinstance(z, int):
#         xlabel = 'x [cm]'
#         ylabel = 'y [cm]'

#     fig, ax = plt.subplots(1, 1, figsize=figsize,facecolor=(1, 1, 1) )
#     ax.set_title(titulo)
#     for carga in Q:
#         q, xq, yq, zq = carga
#         # Different colors for positive and negative charges.
#         if q>0:
#             color = 'red'
#         else:
#             color = 'blue'
#         # Check if the charge has to be drawn or not.
#         if isinstance(x, float) or isinstance(x, int):
#             if xq == x:
#                 circ = plt.Circle((yq*100,zq*100), dq*dim, color=color)
#                 ax.add_patch(circ)
#         elif isinstance(y, float) or isinstance(y, int):
#             if yq == y:
#                 circ = plt.Circle((xq*100,zq*100), dq*dim, color=color)
#                 ax.add_patch(circ)
#         elif isinstance(z, float) or isinstance(z, int):
#             if zq == z:
#                 circ = plt.Circle((xq*100,yq*100), dq*dim, color=color)
#                 ax.add_patch(circ)

#     if EF:
#         CS2 = ax.contour(X, Y, Vmat, levels = levels, colors = 'red', alpha=0.4)
#         E = np.gradient(-1*Vmat)
#         ax.streamplot(X, Y, E[1], E[0], linewidth=1, cmap=plt.cm.inferno,
#               density=density, arrowstyle='->', arrowsize=1.5)
#     else:
#         CS2 = ax.contour(X, Y, Vmat, levels = levels, colors = 'red', alpha=1)
    
#     ax.clabel(CS2, inline=True, fmt=fmtV, fontsize=10)

#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid()
#     plt.show()

#     # return Vmat
