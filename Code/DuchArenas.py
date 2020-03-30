import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
import random
from plot_generators import *


class DA2communityClassifier():
    def __init__(self,graph,B=None,m=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.Q_history=[self.Q]
        self.m=np.sum(self.k)/2
        self.B=self.A-self.k.dot(self.k.T)/(2*self.m)
        self.done=False
        self.s=None
        self.G_positive=None
        self.G_negative=None

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

    def fit(self,eps=1e-4,maxiter=100,node_shift=1,verbose=False):
        Q=self.Q+2*eps
        self.count=0
        nodes=np.array(self.G.nodes)
        index_nodes=np.arange(0,len(self.G.nodes))
        index_nodes_random=np.random.permutation(index_nodes)
        index_nodes_positive=list(index_nodes_random[:len(self.G.nodes)//2])
        index_nodes_negative=list(index_nodes_random[len(self.G.nodes)//2:])
        self.G_positive=self.G.subgraph(nodes[index_nodes_positive])
        self.G_negative=self.G.subgraph(nodes[index_nodes_negative])
        a_positive=1-np.sum(to_numpy_matrix(self.G_negative))/self.m #fraction of edges connected to the positive cluster
        a_negative=1-np.sum(to_numpy_matrix(self.G_positive))/self.m
        category={node:1 if i in index_nodes_positive else -1 for i,node in enumerate(self.G)}
        self.fitness={node:0 for node in self.G.nodes} #fitness between 1 and -1
        while True:
            # For each node
            for i,node in enumerate(self.G.nodes):
                # Compute fitness
                k=self.k[i] #degree of the node
                if category[node]==1:
                    k_com=self.G_positive.degree[node] # degree of the node within the community
                    a=a_positive #fractions of edges connected to the cluster the nodes belong to
                else:
                    k_com=self.G_negative.degree[node]
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
            self.G_positive=self.G.subgraph(nodes[index_nodes_positive])
            self.G_negative=self.G.subgraph(nodes[index_nodes_negative])
            a_positive=1-np.sum(to_numpy_matrix(self.G_negative))/self.m
            a_negative=1-np.sum(to_numpy_matrix(self.G_positive))/self.m
            #Compute Q
            self.Q=Q
            self.s=np.array([1 if i in index_nodes_positive else -1 for i in range(len(self.G.nodes))])
            Q=np.einsum("i,ij,j",self.s,self.B,self.s)/(4*self.m)
            # Q=self.modularity(self.G_positive,self.G_negative)
            self.Q_history.append(Q)
            self.count+=1
            if self.count%10==0 and verbose:
                print("iteration %d"%(self.count))
            if abs(Q-self.Q)<eps or self.count>maxiter:
                if self.Q<=0:
                    self.done=True
                break            
        #Append final result
        # print("Calculated modularity %.3f"%(self.modularity(self.G_positive,self.G_negative)))
        for node in self.G.nodes:
            self.category[node].append(category[node])
