
import networkx as nx
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
from collections import defaultdict

class SP2CcommunityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
        self.sc = SpectralClustering(2, affinity='precomputed')
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.s=None
        self.G_positive=None
        self.G_negative=None
        self.done=False

    def fit(self):
      self.sc.fit(self.A)
      #self.sc.labels_
      rows=list(zip(self.sc.labels_,list(self.G.nodes)))
      d = defaultdict(list)
      for k, v in rows: 
        d[k].append(v)
      partitions=list(d.values())
      ll=[]
      for i in self.sc.labels_:
        label=[2*int(h)-1 for h in list(bin(i)[2:])]
        ll.append(label)
      self.category=dict(zip(list(self.G.nodes),ll))
      self.s=np.array([self.category[node][0] for node in self.G.nodes])
      nodes=np.array(self.G.nodes)
      self.G_positive=self.G.subgraph(nodes[self.s==1])
      self.G_negative=self.G.subgraph(nodes[self.s==-1])
      self.Q=np.einsum("i,ij,j",self.s,self.B,self.s)/(4*self.m)
      if self.Q<0:
        self.done=True
