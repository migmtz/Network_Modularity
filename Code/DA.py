import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm

G=networkx.generators.karate_club_graph()

class DA2communityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.Q=0
        self.category={node:[] for node in graph.nodes}

    
    def argmin(self,d):
        if not d: 
            return None
        # import ipdb; ipdb.set_trace()
        min_val = min(d.values())
        return [k for k in d if d[k] == min_val][0]

    def fit(self,eps=1,maxiter=1000):
        Q=self.Q+2*eps
        count=0
        nodes=np.array(self.G.nodes)
        index_nodes=np.arange(0,len(self.G.nodes))
        index_nodes_random=np.random.permutation(index_nodes)
        index_nodes_positive=list(index_nodes_random[:len(self.G.nodes)//2])
        index_nodes_negative=list(index_nodes_random[len(self.G.nodes)//2:])
        graph_positive=self.G.subgraph(nodes[index_nodes_positive])
        graph_negative=self.G.subgraph(nodes[index_nodes_negative])
        category={node:1 if i in index_nodes_positive else -1 for i,node in enumerate(self.G)}
        self.fitness={node:0 for node in self.G}
        while True:
            # For each node
            for i,node in enumerate(self.G.nodes):
                #Compute fitness
                k=self.k[i] #degree of the node

                if category[node]==1:
                    k_com=graph_positive.degree[node] #degree of the node within the community
                else:
                    k_com=graph_negative.degree[node]
                self.fitness[node]=2*k_com/k-1

                #Move the node with lowest fitness to the opposite community
                node_min=self.argmin(self.fitness)
                arg_node_min=list(index_nodes).index(node_min)
                if category[node_min]==1:
                    index_nodes_positive.remove(arg_node_min)
                    index_nodes_negative.append(arg_node_min)
                    category[node_min]=-1
                else:
                    index_nodes_negative.remove(arg_node_min)
                    index_nodes_positive.append(arg_node_min)
                    category[node_min]=1

                #Recompute the subgraphs parameters
                graph_positive=self.G.subgraph(nodes[index_nodes_positive])
                graph_negative=self.G.subgraph(nodes[index_nodes_negative])
                    
            #Compute Q
            self.Q=Q
            B=self.A-np.diag(self.k)
            s=np.array([1 if i in index_nodes_positive else -1 for i in range(len(self.G.nodes))])
            Q=np.einsum("i,ij,j",s,B,s)/(4*self.m)
            count+=1
            if abs(Q-self.Q)<eps or count>maxiter:
                break
        #Append final result
        for node in self.G.nodes:
            self.category[node].append(category[node])

# Multiclass classifier to test

clf=DA2communityClassifier(G)
clf.fit()
print("Q-value %f"%(clf.Q))
print("categories %s"%(str(clf.category)))