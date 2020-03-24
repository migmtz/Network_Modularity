import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
# --------------------------- Modularity evolution --------------------------- #

def plot_Q(graph,eps=1e-3,maxQ=False):
    """
        Build a classifier stopping at each level N, compute the corresponding modularity
    """
    q1=0
    q2=q1+2*eps
    Q_results=[0]
    i=1
    while q2-q1>eps:
       clfN=NCommunityClassifier(graph,level=i)
       clfN.fit()
       q1=q2
       q2=clfN.Q 
       Q_results.append(q2)
       i+=1
    plt.plot(np.arange(0,i),Q_results)
    plt.xlabel("Division level")
    plt.ylabel("Modularity")
    plt.show()
    if maxQ:
        return q2

# ----------------------------- Plot communities ----------------------------- #

# import cm
def plot_communities(G,clf):
    # Labelize lists
    dict_aux = {}
    dict_labels = {}
    i = -1
    for key,val in clf.category.items():
        if dict_aux.get(tuple(val)) is None:
            i += 1
        a = dict_aux.setdefault(tuple(val),i)
        dict_labels.setdefault(key,a)
    print(dict_aux)
    # Plot parameters
    pos = networkx.kamada_kawai_layout(G)
    rainbow = cm.rainbow(np.linspace(0,1,len(dict_aux)))
    
    plt.figure()
    for k in range(len(dict_aux)):
        nodes = [i for i in dict_labels.keys() if dict_labels[i] == k]
        networkx.draw_networkx_nodes(G,pos,
                                nodelist = nodes,
                                node_color =rainbow[k].reshape(1,4),
                                node_size=200,
                                node_shape = 'o',
                                label = str(k),
                                alpha=0.8)

    
    networkx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    plt.legend()
    plt.show()


# ----------------------------- Plot with eigen ------------------------------ #

def a_b(list,q):
    diff = max(list) - min(list)
    a = (1-q)/(diff)
    b = (diff -(1-q)*max(list))/(diff)
    return([min(a*l + b,1) for l in list])


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        
def plot_communities_eigen(G,clf): # For now only with two communities
    # Labelize lists
    dict_aux = {}
    dict_labels = {}
    i = -1
    for key,val in clf.category.items():
        if dict_aux.get(tuple(val)) is None:
            i += 1
        a = dict_aux.setdefault(tuple(val),i)
        dict_labels.setdefault(key,a)
    print(dict_aux)

    # Plot parameters
    pos = networkx.kamada_kawai_layout(G)
    rainbow = cm.rainbow(np.linspace(0,1,len(dict_aux)))
    gradient = np.abs(clf.leading_eigenvector)
    
    plt.figure()
    aux = 1
    for k in range(len(dict_aux)):
        nodes = [i for i in dict_labels.keys() if dict_labels[i] == k]
        grad = [np.abs(gradient[i]) for i in dict_labels.keys() if dict_labels[i] == k]
        #grad = (grad+max(grad)-2*min(grad))/(2*(max(grad)-min(grad)))
        grad = a_b(grad,1/9)
        aux = rainbow[k].reshape(4,1).repeat(len(grad),axis=1)
        print(lighten_color(rainbow[k],0.3))
        col = (grad*aux).T
        networkx.draw_networkx_nodes(G,pos,
                                nodelist = nodes,
                                node_color = [lighten_color(rainbow[k],p) for p in grad],
                                node_size=200,
                                node_shape = 'o',
                                label = str(k),
                                alpha=1)
        aux = len(nodes)*2
        print(aux, rainbow.shape)
        aux += 1
    
    networkx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    plt.legend()
    plt.show()

# plot_communities_eigen(G,clf)
