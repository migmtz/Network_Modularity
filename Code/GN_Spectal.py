
import networkx as nx
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
from community_classifiers import plot_communities

G=nx.generators.karate_club_graph()


#GN method two cmmunity
class GN2communityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.communities_generator = community.girvan_newman(graph)
        self.Q=0
        self.category={node:[] for node in graph.nodes}

    def fit(self):
      partitions=next(self.communities_generator)
      B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
      s=np.array([1 if i in partitions[0] else -1 for i in range(self.G.number_of_nodes())])
      self.Q=np.einsum("i,ij,j",s,B,s)/(4*self.m)
      self.category=dict(zip(list(self.G.nodes), [[i] for i in s]))


clf_GN=GN2communityClassifier(G)
clf_GN.fit()
print("Q-value for GN %f"%(clf_GN.Q))
print("categories GN %s"%(str(clf_GN.category)))


#spectral clustering two community
class SPC2communityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.Q=0
        self.category={node:[] for node in graph.nodes}

    def fit(self):
      sc = SpectralClustering(2, affinity='precomputed')
      sc.fit(self.A)
      B=self.A-np.dot(self.k,self.k.transpose())/(2*self.m)
      s=np.array([1 if i==1 else -1 for i in sc.labels_])
      self.Q=np.einsum("i,ij,j",s,B,s)/(4*self.m)
      self.category=dict(zip(list(self.G.nodes), [[i] for i in s]))


clf_spc=SPC2communityClassifier(G)
clf_spc.fit()
print("Q-value for spectral clustering %f"%(clf_spc.Q))
print("categories spectral clustering %s"%(str(clf_spc.category)))

plot_communities(G,clf_spc)