import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

class Newman2CommunityClassifier():
    
    def __init__(self,graph,B=None,m=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
        self.leading_eigenvector=None
        self.category={node:[] for node in self.G.nodes}
        self.s=None
        self.done=False
        self.Q=0
        self.G_positive=None
        self.G_negative=None
    
    def fit(self):
        vals,vecs=eig(self.B)
        self.leading_eigenvector=np.ravel(vecs[:,np.argmax(vals)])
        self.s=np.array([1 if v>=0 else -1 for v in self.leading_eigenvector])
        for i,node in enumerate(self.G.nodes):
            self.category[node].append(self.s[i])
        nodes=np.array(self.G.nodes)
        self.G_positive=self.G.subgraph(nodes[self.s==1])
        self.G_negative=self.G.subgraph(nodes[self.s==-1])
        self.Q=np.einsum("i,ij,j",self.s,self.B,self.s)/(4*self.m)
        if self.Q<=0 or np.max(self.leading_eigenvector)*np.min(self.leading_eigenvector)>0: #All elements of the same sign or negative modularity
            self.done=True
    