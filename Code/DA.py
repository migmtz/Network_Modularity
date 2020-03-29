import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
import random
from plot_generators import *
np.random.seed(1) 
# from community_classifiers import plot_communities
# G=networkx.generators.karate_club_graph()
G = networkx.read_gml('./data/polbooks.gml')

class DA2communityClassifier():
    def __init__(self,graph,B=None,m=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.Q_history=[self.Q]
        if m is not None:
            self.m=m
        else:
            self.m=np.sum(self.k)/2
        if B is not None:
            self.B=B
        else:
            self.B=self.A-self.k.dot(self.k.T)/(2*self.m)
        self.done=False

    def modularity(self,graph_positive,graph_negative):
        m_positive=np.sum(to_numpy_matrix(graph_positive))/2
        m_negative=np.sum(to_numpy_matrix(graph_negative))/2
        m=np.sum(to_numpy_matrix(self.G))/2
        return (3*m*(m_positive+m_negative)-2*m**2-(m_positive**2+m_negative**2))/m**2
    
    def argmin(self,d):
        """
        Returns the key corresponding to the minimal value in a dictionnary
        """
        if not d: 
            return None
        # import ipdb; ipdb.set_trace()
        min_val = min(d.values())
        return [k for k in d if d[k] == min_val][0]

    def fit(self,eps=1e-5,maxiter=100,node_shift=3):
        Q=self.Q+2*eps
        self.count=0
        nodes=np.array(self.G.nodes)
        index_nodes=np.arange(0,len(self.G.nodes))
        index_nodes_random=np.random.permutation(index_nodes)
        index_nodes_positive=list(index_nodes_random[:len(self.G.nodes)//2])
        index_nodes_negative=list(index_nodes_random[len(self.G.nodes)//2:])
        graph_positive=self.G.subgraph(nodes[index_nodes_positive])
        graph_negative=self.G.subgraph(nodes[index_nodes_negative])
        a_positive=1-np.sum(to_numpy_matrix(graph_negative))/self.m #fraction of edges connected to the positive cluster
        a_negative=1-np.sum(to_numpy_matrix(graph_positive))/self.m
        category={node:1 if i in index_nodes_positive else -1 for i,node in enumerate(self.G)}
        self.fitness={node:0 for node in self.G.nodes} #fitness between 1 and -1
        while True:
            # For each node
            for i,node in enumerate(self.G.nodes):
                # Compute fitness
                k=self.k[i] #degree of the node
                if category[node]==1:
                    k_com=graph_positive.degree[node] # degree of the node within the community
                    a=a_positive #fractions of edges connected to the cluster the nodes belong to
                else:
                    k_com=graph_negative.degree[node]
                    a=a_negative
                self.fitness[node]=k_com/k-a
            # Move the node_shift nodes with lowest fitness to the opposite community
            candidates=self.fitness.copy()
            for j in range(node_shift):
                node_min=self.argmin(candidates)
                index_node_min=list(nodes).index(node_min)
                # print(index_node_min)
                if category[node_min]==1:
                    index_nodes_positive.remove(index_node_min)
                    index_nodes_negative.append(index_node_min)
                    category[node_min]=-1
                else:
                    index_nodes_negative.remove(index_node_min)
                    index_nodes_positive.append(index_node_min)
                    category[node_min]=1
                del candidates[node_min]
            # Recompute the subgraphs parameters
            graph_positive=self.G.subgraph(nodes[index_nodes_positive])
            graph_negative=self.G.subgraph(nodes[index_nodes_negative])
            a_positive=1-np.sum(to_numpy_matrix(graph_negative))/self.m
            a_negative=1-np.sum(to_numpy_matrix(graph_positive))/self.m
            #Compute Q
            self.Q=Q
            s=np.array([1 if i in index_nodes_positive else -1 for i in range(len(self.G.nodes))])
            Q=np.einsum("i,ij,j",s,self.B,s)/(4*self.m)
            self.Q_history.append(Q)
            self.count+=1
            if self.count%10==0:
                print("iteration %d"%(self.count))
            if abs(Q-self.Q)<eps or self.count>maxiter:
                if self.Q<=0:
                    self.done=True
                break            
        #Append final result
        print("Calculated modularity %.3f"%(self.modularity(graph_positive,graph_negative)))
        for node in self.G.nodes:
            self.category[node].append(category[node])

# Multiclass classifier to test
class DANcommunityClassifier(DA2communityClassifier):

    def __init__(self,graph,B=None,category=None,Nmax=None):
        super().__init__(graph,B)
        self.Nmax=Nmax
        self.Q=0
        self.N=1
    
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
        clf=DA2communityClassifier(graph,B,self.m)
        clf.fit()
        if clf.done or self.Nmax==0:
            # If it is an undivisible graph, do not return any classification and terminate the fitting operation 
            return None
        else:
            # Otherwise, assign each node of the considered graph to its category
            if self.Nmax:
                self.Nmax-=1
            self.Q+=clf.Q
            self.N+=1 
            index_positive=[]
            index_negative=[]
            nodes_positive=[]
            nodes_negative=[]
            for i,node in enumerate(graph.nodes):
                if clf.category[node]==[1]:
                    self.category[node].append(1)
                    index_positive.append(i)
                    nodes_positive.append(node)
                else:
                    self.category[node].append(-1)
                    index_negative.append(i)
                    nodes_negative.append(node)
            #Iterate the division on the two subgraphs
            nodes=np.array(graph.nodes)
            index=np.arange(0,len(nodes))
            subgraph_positive=graph.subgraph(nodes_positive)
            subgraph_negative=graph.subgraph(nodes_negative)
            B_positive=self.Beq(index_positive)
            B_negative=self.Beq(index_negative)
            self.fit(subgraph_positive,B_positive,category)
            self.fit(subgraph_negative,B_negative,category)



clf=DA2communityClassifier(G)
clf.fit()
print("Q-value %f"%(clf.Q))
print("categories %s"%(str(clf.category)))
print("count %d"%(clf.count))
plot_communities(G,clf)

# clfN=DANcommunityClassifier(G,Nmax=10)
# clfN.fit()
# print("Q-value %f"%(clfN.Q))
# print("categories %s"%(str(clfN.category)))
# plot_communities(G,clfN)