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


# 20250611
def Ef(x, y, z, Q):
    """Calcula las componentes del campo eléctrico en N/C producido por un sistema de
    cargas puntuales.

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


# 20250611
def V(x,y,z,Q):
    """Calcula potencial eléctrico en Volt en la posición (x,y,z), con una distribución de
    cargas puntuales Q.

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


# 20250613
# TODO: add more control over plotting parameters.
# Add examples in the docstring.
def plotEf(Q, **params):
    """
    Muestra las líneas de campo eléctrico en 2D, para un sistema de cargas puntuales.

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
    axs : matplotlib.axes.Axes
        Objeto axes donde mostrar las líneas de campo. Si está vacío,
        se genera una figura nueva.    
    start : array
        Array para alimentar a Start_points de streamplot, en el formato
        [[x1,x2,x3,...],[y1,y2,y3,...]]

    *Además de los parámetros de matplotlib y streamplot, por ejemplo:*
    figsize : tuple
    title : string
    """

    dx = params.get('dx', 5)
    dy = params.get('dy', dx)
    w = params.get('w', 100)
    axs = params.get('axs', False)
    start = params.get('start', [])

    figsize = params.get('figsize', (5,5))
    title = params.get('title', 'Líneas de campo')
    linewidth = params.get('linewidth', 0.4)
    density = params.get('density', 0.7)

    # Convirtiendo w a número complejo se incluye el extremo del intervalo en mgrid.
    w = w * 1j
    Y, X = np.mgrid[-dx:dx:w, -dy:dy:w]
    Z = 0*X

    Ei, Ej, Ek = Ef(X,Y,Z,Q)

    if not axs:
        fig, axs = plt.subplots(1, 1, figsize=figsize)
    if len(start) == 0:
        strm = axs.streamplot(X, Y, Ei, Ej, color='b',
             linewidth=linewidth, density=density)
    else:
        strm = axs.streamplot(X, Y, Ei, Ej, color='b', linewidth=linewidth, density=density, start_points=np.array(start).T)

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


# 20250613
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
        Limites de los ejes: [xmin, xmax, ymin, ymax]
    aumento : float
        Factor para cambiar el tamaño de la figura. Mantiene
        la relación entre ejes.
    scale : float
        Regula la longitud de las flechas.
    lineas : float
        Si es True, se grafican las líneas de campo eléctrico.
    contribuciones : boolean
        Graficar los vectores producidos por cada carga, además del resultante.
    normalizados : boolean
        Normalizar todos los vectores. Utilizar cuando las longitudes de las flechas
        resultan incómodas, para solo visualizar direcciones y sentidos.
    conectores : boolean
        Muestra una recta punteada desde la carga hasta el punto campo.
    linewidth : float
        Ancho de las líneas que muestran la dirección.
    
    *Además de los parámetros de matplotlib y quiver, por ejemplo:*
    length : float
    figsize : tuple (no usar, se controla con limites y aumento)
    title : string
    """

    figsize = params.get('figsize', (5,5))
    title = params.get('title', "Algunos vectores de campo eléctrico")
    scale = params.get('scale', 1)
    lineas = params.get('lineas', False)
    contribuciones = params.get('contribuciones', False)
    normalizados = params.get('normalizados', False)
    conectores = params.get('conectores', False)
    linewidth = params.get('linewidth', 0.5)
    aumento = params.get('aumento', 1)

    xmin, xmax, ymin, ymax = 0,0,0,0
    x_pos = []
    y_pos = []
    Ei = []
    Ej = []
    color = []

    if contribuciones:
        for q in Q:
            if q[0] > 0:
                cq = 'r'
            else :
                cq = 'g'
            for x in X:
                Eii, Ejj, Ekk = Ef(x[0],x[1],x[2],[q])
                x_pos = np.concatenate((x_pos,x[0]), axis=None)
                y_pos = np.concatenate((y_pos,x[1]), axis=None)
                if normalizados:
                    # N = np.sqrt(Eii**2 + Ejj**2)*1.5
                    N = np.sqrt(Eii**2 + Ejj**2)
                else:
                    N = 1
                Ei = np.concatenate((Ei, Eii/N), axis=None)
                Ej = np.concatenate((Ej, Ejj/N), axis=None)
                color = color + [cq]

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
        if normalizados:
            # N = np.sqrt(Eii**2 + Ejj**2)*1.5
            N = np.sqrt(Eii**2 + Ejj**2)
        else:
            N = 1
        Ei = np.concatenate((Ei, Eii/N), axis=None)
        Ej = np.concatenate((Ej, Ejj/N), axis=None)
        color = color + ['k']

    # Creating plot
    fig, ax = plt.subplots(figsize = figsize)
    ax.quiver(x_pos, y_pos, Ei, Ej, angles='xy', scale_units='xy', scale=scale, color=color)
    # Experimentar con el parámetro width para el ancho de las flechas.
    # ax.quiver(x_pos, y_pos, Ei, Ej, angles='xy', scale_units='xy', scale=scale,
        # width=0.00075*(xmax-xmin), color=color)

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

    if conectores:
        for q in Q:
            qq, xq, yq, zq = q
            for x in X:
                ax.plot([xq,x[0]], [yq,x[1]], color='b', linewidth=linewidth, linestyle='dashed')

    # ax.set_title(title)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    # Se expanden los límites automáticos:
    stretch = 0.15
    xmax2 = xmax + (xmax - xmin)*stretch
    xmin2 = xmin - (xmax - xmin)*stretch
    ymax2 = ymax + (ymax - ymin)*stretch
    ymin2 = ymin - (ymax - ymin)*stretch

    limites = params.get('limites', [xmin2,xmax2,ymin2,ymax2])
    ax.axis(limites)
    ax.set_title(title)
    if normalizados:
        ax.text(limites[0] + 0.5, limites[2] + 0.1, "Vectores normalizados", fontsize=12)

    if lineas:
        # Lista de puntos para que las líneas de campo
        # pasen por los vectores graficados.
        a = []
        b = []
        for x in X:
            a = a + [x[0]]
            b = b + [x[1]]
        start = [a,b]
        plotEf(Q, start=start, axs=ax)
        
    # plt.axis('equal')
    fig.set_size_inches(((limites[1]-limites[0])*aumento, (limites[3]-limites[2])*aumento))
    plt.show()
    # plt.close()


# 20240819
#TODO: agregar contribuciones.
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

