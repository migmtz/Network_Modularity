import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm

# --------------------------- Generate sample graph -------------------------- #

G=networkx.generators.karate_club_graph()
# G = networkx.read_gml('./data/polbooks.gml')


print("Matrice d'adjacence")
print(to_numpy_matrix(G))
print("Vecteurs des degres")
print(np.sum(to_numpy_matrix(G),axis=1))

# --------------- Create tree object to store modularity values -------------- #

class Tree():
    def __init__(self,root=0,left=None,right=None):
        self.root=root
        self.left=left
        self.right=left

    def isempty(self):
        return self.root==0

    def push_left(self,value=None):
        self.left=Tree(value)
    
    def push_right(self,value=None):
        self.right=Tree(value)
    
    def sum_level(self):
        pass


# ----------------------- 2-Communities classifier ---------------------- #

class TwoCommunityClassifier():
    
    def __init__(self,graph,B=None,m=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        if m is None:
            self.m=np.sum(self.k)/2
        else:
            self.m=m
        if B is None:
            self.B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
        else:
            self.B=B
        self.leading_eigenvector=None
        self.category={node:[] for node in self.G.nodes}
        self.done=False

    
    def fit(self,eps=0.5,max_iteration=1000):
        vals,vecs=eig(self.B)
        self.leading_eigenvector=np.ravel(vecs[:,np.argmax(vals)])
        s=[1 if v>=0 else -1 for v in self.leading_eigenvector]
        for i,node in enumerate(self.G.nodes):
            self.category[node].append(s[i])
        self.Q=np.einsum("i,ij,j",s,self.B,s)/(4*self.m)
        if self.Q<=0 or np.max(self.leading_eigenvector)*np.min(self.leading_eigenvector)>0: #All elements of the same sign or negative modularity
            self.done=True
    

# ------------------------- N-Communities classifier ------------------------- #

class NCommunityClassifier(TwoCommunityClassifier):
    def __init__(self,graph,B=None,category=None,level=None):
        super().__init__(graph,B)
        self.level=level
        self.Q=0
        self.N=0
    
    def Beq(self,nodes):
        #compute the equivalent matrix Beq
        Beq=self.B[nodes,:][:,nodes]
        Beq-=np.diagonal(np.sum(Beq,axis=1))
        return Beq

    def fit(self,graph=None,B=None,category=None):

        if graph is None:
            graph=self.G
        if category:
            self.category=category
        # The first step is to attempt a split on the considered graph.
        clf=TwoCommunityClassifier(graph,B,self.m)
        clf.fit()
        if clf.done or self.level==0:
            # If it is an undivisible graph, do not return any classification and terminate the fitting operation 
            return None
        else:
            # Otherwise, assign each node of the considered graph to its category
            if self.level:
                self.level-=1
            self.Q+=clf.Q
            self.N+=2  
            for i,node in enumerate(graph.nodes):
                self.category[node].append(1 if clf.leading_eigenvector[i]>=0 else -1)
            #Iterate the division on the two subgraphs
            nodes=np.arange(len(clf.leading_eigenvector))
            nodes_positive=nodes[clf.leading_eigenvector>=0]
            nodes_negative=nodes[clf.leading_eigenvector<0]
            subgraph_positive=graph.subgraph(nodes_positive)
            subgraph_negative=graph.subgraph(nodes_negative)
            B_positive=self.Beq(nodes_positive)
            B_negative=self.Beq(nodes_negative)
            self.fit(subgraph_positive,B_positive,category)
            self.fit(subgraph_negative,B_negative,category)


# ---------------------------------------------------------------------------- #
#                                    Results                                   #
# ---------------------------------------------------------------------------- #

clf=TwoCommunityClassifier(G)
clf.fit()
print("Two communities modularity: %f"%(clf.Q))


clfN=NCommunityClassifier(G)  
clfN.fit()  
print("N-communities modularity:%f"%(clfN.Q))
print("Number of communities found:%d"%(clfN.N))


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

plot_Q(G)

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

plot_communities(G,clf)
plot_communities(G,clfN)

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
