import networkx as nx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
from plot_generators import *
from networkx.algorithms import community


#GN Algorithm for 2 or more cmmunities
class GN2communityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
        self.communities_generator = community.girvan_newman(graph)
        self.Q=0
        self.nb_iter=1
        self.category={}
        self.s=None
        self.G_positive=None
        self.G_negative=None
        self.done=False

    def fit(self):
      for i in range(self.nb_iter):
        partitions=next(self.communities_generator)
      for i,l in enumerate(partitions):
        labels=[[2*int(h)-1 for h in list(bin(i)[2:])]]*len(l)
        self.category.update(dict(zip(l,labels)))
      self.s=np.array([self.category[node][0] for node in self.G.nodes])
      nodes=np.array(self.G.nodes)
      self.G_positive=self.G.subgraph(nodes[self.s==1])
      self.G_negative=self.G.subgraph(nodes[self.s==-1])
      self.Q=np.einsum("i,ij,j",self.s,self.B,self.s)/(4*self.m)
      if self.Q<0:
        self.done=True