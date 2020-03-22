import networkx as nx
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
from sklearn.cluster import SpectralClustering

G=nx.generators.karate_club_graph()

#spectral clustering two community
class SPC2communityClassifier():
    def __init__(self,graph,B=None,m=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        if m is None:
          self.m=np.sum(self.k)/2
        else:
          self.m=m
        if B is None:
          self.B=self.A-(self.k.dot(self.k.T))/(2*self.m)
        else:
          self.B=B
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.done=False

    def fit(self):
      sc = SpectralClustering(2, affinity='precomputed')
      sc.fit(self.A)
      s=np.array([1 if i==1 else -1 for i in sc.labels_])
      self.Q=np.einsum("i,ij,j",s,self.B,s)/(4*self.m)
      self.category=dict(zip(list(self.G.nodes), [[i] for i in s]))
      if self.Q<=0:
        self.done=True

clf_spc=SPC2communityClassifier(G)
clf_spc.fit()
print("modularity for spectral clustering %f"%(clf_spc.Q))
print("categories spectral clustering %s"%(str(clf_spc.category)))


class SPCNCommunityClassifier(SPC2communityClassifier):
    def __init__(self,graph,B=None,category=None,level=None):
        super().__init__(graph,B)
        self.level=level
        self.Q=0
        self.N=1
    
    def Beq(self,nodes):
        #compute the equivalent matrix Beq
        Beq=self.B[nodes,:][:,nodes]
        # import ipdb; ipdb.set_trace()
        Beq-=np.diagonal(np.sum(Beq,axis=1))
        return Beq

    def fit(self,graph=None,B=None,category=None):

        if graph is None:
            graph=self.G
        if category:
            self.category=category
        # The first step is to attempt a split on the considered graph.
        clf=SPC2communityClassifier(graph,B,self.m)
        clf.fit()
        if clf.done or self.level==0:
            # If it is an undivisible graph, do not return any classification and terminate the fitting operation 
            return None
        else:
            
            # Otherwise, assign each node of the considered graph to its category
            if self.level:
                self.level-=1
            self.Q+=clf.Q
            self.N+=1
            label=np.array([i[0] for i in clf.category.values()]) 
            for i,node in enumerate(graph.nodes):
                self.category[node].append(1 if label[i]>=0 else -1)
            
            #Iterate the division on the two subgraphs
            nodes=np.array(graph.nodes)
            index=np.arange(0,len(nodes))
            nodes_positive=nodes[label>=0]
            nodes_negative=nodes[label<0]
            index_positive=index[label>=0]
            index_negative=index[label<0]
            subgraph_positive=graph.subgraph(nodes_positive)
            subgraph_negative=graph.subgraph(nodes_negative)
            B_positive=self.Beq(index_positive)
            B_negative=self.Beq(index_negative)
            self.fit(subgraph_positive,B_positive,category)
            self.fit(subgraph_negative,B_negative,category)


clfN_spc=SPCNCommunityClassifier(G,level=3)  
clfN_spc.fit()  
print("N-communities modularity:%f"%(clfN_spc.Q))
print("Number of communities found:%d"%(clfN_spc.N))
print("categories(n) spectral clustering %s"%(str(clfN_spc.category)))
