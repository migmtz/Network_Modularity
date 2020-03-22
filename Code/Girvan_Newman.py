import networkx as nx
import numpy as np
from networkx.algorithms import community


G=nx.generators.karate_club_graph()


#GN method two cmmunity
class GNcommunityClassifier():
    def __init__(self,graph,nb_community=2):
        self.G=graph
        self.communities_generator = community.girvan_newman(graph)
        self.Q=0
        self.nb_iter=nb_community-1
        self.category={}

    def fit(self):
      for i in range(self.nb_iter):
        partitions=next(self.communities_generator)
      self.Q=community.modularity(self.G,partitions)
      dd={}
      for i,l in enumerate(partitions):
        #labels=np.full(len(l),i)
        labels=[[2*int(h)-1 for h in list(bin(i)[2:])]]*len(l)
        dd.update(dict(zip(l,labels)))
      self.category=dd
      

clf_gn_2=GNcommunityClassifier(G,nb_community=2)
clf_gn_2.fit()
print("Modularity GN 2 community: %f"%(clf_gn_2.Q))
print("Categories GN 2 community:  %s"%(str(clf_gn_2.category)))# Attention! the dict is not sorted

clf_gn_3=GNcommunityClassifier(G,nb_community=3)
clf_gn_3.fit()
print("Modularity GN 3 community: %f"%(clf_gn_3.Q))
print("Categories GN 3 community:  %s"%(str(clf_gn_3.category)))
