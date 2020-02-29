import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

# --------------------------- Generate sample graph -------------------------- #

G=networkx.generators.karate_club_graph()

print("Matrice d'adjacence")
print(to_numpy_matrix(G))
print("Vecteurs des degrés")
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
            self.m=np.sum(self.k)
        else:
            self.m=m
        if B is None:
            self.B=self.A-np.dot(self.k,self.k.transpose())/self.m
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
        if category:
            self.category=category
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
        
        clf=TwoCommunityClassifier(graph,B,self.m)
        clf.fit()
        if clf.done or self.level==0:
            # If undivisible graph, do not return any classification and terminate the fitting operation 
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
        