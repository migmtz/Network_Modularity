
import networkx as nx
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
from collections import defaultdict

G=nx.generators.karate_club_graph()

#spectral clustering for 2 or more communities
class SPCcommunityClassifier():
    def __init__(self,graph,nb_community=2):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.sc = SpectralClustering(nb_community, affinity='precomputed')
        self.Q=0
        self.category={node:[] for node in graph.nodes}

    def fit(self):
      self.sc.fit(self.A)
      #self.sc.labels_
      rows=list(zip(self.sc.labels_,list(self.G.nodes)))
      d = defaultdict(list)
      for k, v in rows: 
        d[k].append(v)
      partitions=list(d.values())
      self.Q=community.modularity(self.G,partitions)

      ll=[]
      for i in self.sc.labels_:
        label=[2*int(h)-1 for h in list(bin(i)[2:])]
        ll.append(label)
      self.category=dict(zip(list(self.G.nodes),ll))
      


clf_spc_2=SPCcommunityClassifier(G,nb_community=2)
clf_spc_2.fit()
print("Modularity spectral 2 community %f"%(clf_spc_2.Q))
print("categories spectral 2 community %s"%(str(clf_spc_2.category)))

clf_spc_3=SPCcommunityClassifier(G,nb_community=3)
clf_spc_3.fit()
print("Modularity spectral 3 community %f"%(clf_spc_3.Q))
print("categories spectral 3 community %s"%(str(clf_spc_3.category)))
