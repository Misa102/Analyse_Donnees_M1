#######################################
# biplot
# version 18/11/2019
#######################################
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns

def my_biplot(score=None,coeff=None,coeff_labels=None,score_labels=None,nomx = None,nomy = None):
    
    cmap="Set1"
    xs = score[:,0]
    ys = score[:,1]

    if (len(xs) != len(ys)) : print("Warning ! len(x) != len(y)")
    
    x_c = (xs-xs.mean())/(xs.max() - xs.min())
    y_c = (ys-ys.mean())/(ys.max() - ys.min())
    

    fig = plt.figure(figsize=(6,6),facecolor='w') 
    ax = fig.add_subplot(111)

    # Affichage des points
    ax.scatter(x_c,y_c,cmap=cmap)
    if score_labels is not None :
        n = len(x_c)
     
        for i in range(0,n) :
            
            plt.text(x_c[i]+0.01,y_c[i],score_labels[i])

    
    
    x_circle = np.linspace(-1, 1, 100)
    y_circle = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_circle,y_circle)
    F = X**2 + Y**2 - 1.0
    #fig, ax = plt.subplots()
    plt.contour(X,Y,F,[0])
    p = coeff.shape[0]
    
    for i in range(0,p):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5,head_width=0.05, head_length=0.05)
            
        if coeff_labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, coeff_labels[i], color = 'g', ha = 'center', va = 'center')
    
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.xlabel(nomx)
    plt.ylabel(nomy)
    plt.grid(linestyle='--')
    plt.show()

