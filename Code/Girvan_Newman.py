import networkx as nx
import numpy as np
from networkx.convert_matrix import to_numpy_matrix
from networkx.algorithms import community

G=nx.generators.karate_club_graph()


class GN2communityClassifier():
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
        #self.m=np.sum(self.k)/2
        self.communities_generator = community.girvan_newman(graph)
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.q_b=0
        self.done=False

    def fit(self):
      partitions=next(self.communities_generator)
      #=(self.k.dot(self.k.T))/(2*self.m)
      #B=self.A-kk
      s=np.array([1 if i in partitions[1] else -1 for i in range(self.G.number_of_nodes())])
      self.Q=np.einsum("i,ij,j",s,self.B,s)/(4*self.m)
      self.category=dict(zip(list(self.G.nodes), [[i] for i in s]))
      #self.q_b=community.modularity(self.G,partitions)
      if self.Q<=0:
        self.done=True


clf_GN=GN2communityClassifier(G)
clf_GN.fit()
print("modularity for GN %f"%(clf_GN.Q))
print("categories GN %s"%(str(clf_GN.category)))


class GN_NCommunityClassifier(GN2communityClassifier):
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
        clf=GN2communityClassifier(graph,B,self.m)
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


clfN_gn=GN_NCommunityClassifier(G,level=3)  
clfN_gn.fit()  
print("N-communities modularity:%f"%(clfN_gn.Q))
print("Number of communities found:%d"%(clfN_gn.N))
print("categories GN(n) %s"%(str(clfN_gn.category)))
