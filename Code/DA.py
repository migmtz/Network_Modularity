import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
import random

np.random.seed(0)
# from community_classifiers import plot_communities
G=networkx.generators.karate_club_graph()


class DA2communityClassifier():
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)/2
        self.Q=0
        self.category={node:[] for node in graph.nodes}
        self.Q_history=[self.Q]
    
    def argmin(self,d):
        """
        Returns the key corresponding to the minimal value in a dictionnary
        """
        if not d: 
            return None
        # import ipdb; ipdb.set_trace()
        min_val = min(d.values())
        return [k for k in d if d[k] == min_val][0]

    def fit(self,eps=1e-5,maxiter=1000,node_shift=3):
        Q=self.Q+2*eps
        self.count=0
        nodes=np.array(self.G.nodes)
        index_nodes=np.arange(0,len(self.G.nodes))
        index_nodes_random=np.random.permutation(index_nodes)
        index_nodes_positive=list(index_nodes_random[:len(self.G.nodes)//2])
        index_nodes_negative=list(index_nodes_random[len(self.G.nodes)//2:])
        graph_positive=self.G.subgraph(nodes[index_nodes_positive])
        graph_negative=self.G.subgraph(nodes[index_nodes_negative])
        category={node:1 if i in index_nodes_positive else -1 for i,node in enumerate(self.G)}
        self.fitness={node:0 for node in self.G.nodes} #fitness between 1 and -1
        while True:
            # For each node
            for i,node in enumerate(self.G.nodes):
                # Compute fitness
                k=self.k[i] #degree of the node
                if category[node]==1:
                    k_com=graph_positive.degree[node] # degree of the node within the community
                else:
                    k_com=graph_negative.degree[node] 
                self.fitness[node]=2*k_com/k-1
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
            #Compute Q
            self.Q=Q
            B=self.A-self.k.dot(self.k.T)/(2*self.m)
            s=np.array([1 if i in index_nodes_positive else -1 for i in range(len(self.G.nodes))])
            Q=np.einsum("i,ij,j",s,B,s)/(4*self.m)
            self.Q_history.append(Q)
            self.count+=1
            if abs(Q-self.Q)<eps or self.count>maxiter:
                break
        #Append final result
        for node in self.G.nodes:
            self.category[node].append(category[node])

def plot_communities(G,clf):
    # Labelize lists
    dict_aux = {}
    dict_labels = {}
    i = -1
    for key,val in clf.category.items():
        if dict_aux.get(tuple(val)) is None:
            i += 1
        a = dict_aux.setdefault(tuple(val),i)
        dict_labels.setdefault(key,a)
    print(dict_aux)
    # Plot parameters
    pos = networkx.kamada_kawai_layout(G)
    rainbow = cm.rainbow(np.linspace(0,1,len(dict_aux)))
    
    plt.figure()
    for k in range(len(dict_aux)):
        nodes = [i for i in dict_labels.keys() if dict_labels[i] == k]
        networkx.draw_networkx_nodes(G,pos,
                                nodelist = nodes,
                                node_color =rainbow[k].reshape(1,4),
                                node_size=200,
                                node_shape = 'o',
                                label = str(k),
                                alpha=0.8)

    
    networkx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    plt.legend()
    plt.show()


# Multiclass classifier to test

clf=DA2communityClassifier(G)
clf.fit()
print("Q-value %f"%(clf.Q))
print("categories %s"%(str(clf.category)))
print("count %d"%(clf.count))
# plot_communities(G,clf)