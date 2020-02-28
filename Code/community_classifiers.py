import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
from scipy.sparse import csc_matrix

# --------------------------- Generate sample graph -------------------------- #

G=networkx.generators.karate_club_graph()

print("Matrice d'adjacence")
print(to_numpy_matrix(G))
print("Vecteurs des degrés")
print(np.sum(to_numpy_matrix(G),axis=1))

class Tree():
    def __init__(self,root=0,left=None,right=None):
        self.root=root
        self.left=left
        self.right=left

    def isempty(self):
        return self.root==0

    def push_left(self,value=None):
        self.left=Tree(value)
    
    def push_right(self,value=None):
        self.right=Tree(value)

t=Tree()
# import ipdb; ipdb.set_trace()
    # def sum_level(self,level): 
    # To be completed later
    #     if level==0:
    #         return self.root
    #     # elif self.root
    #     else:
    #         if self.left is None:
    #             return self.root+self.self.left.sum_level(level-1)+self.right.sum_level(level-1)
            
    #         return self..sum_level()


# ----------------------- 2-Communities classifier ---------------------- #

class TwoCommunityClassifier():
    
    def __init__(self,graph,B=None,category=None):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)
        if B is None:
            self.B=self.A-np.dot(self.k,self.k.transpose())/self.m
        else:
            self.B=B
        self.leading_eigenvector=None
        if category is None:
            self.category={node:[] for node in self.G.nodes}
        else:
            self.category=category
        # self.Q=
        self.done=False

    
    def fit(self,eps=0.5,max_iteration=1000):
        # Compute the simple power method (TBC, does not converge)

        #Alternatively, use built in function for eigenvalues
        # import ipdb; ipdb.set_trace()
        vals,vecs=eig(self.B)
        self.leading_eigenvector=np.ravel(vecs[:,np.argmax(vals)])
        s=[1 if v>=0 else -1 for v in self.leading_eigenvector]
        for i,node in enumerate(self.G.nodes):
            self.category[node].append(s[i])
        self.Q=np.einsum("i,ij,j",s,self.B,s)/(4*self.m)
        if self.Q>=0: #All elements of the same sign
            self.done=True
    

# ------------------------- N-Communities classifier ------------------------- #

# The idea is to apply the previous algorithm recursively each community until convergence of the process.
class NCommunityClassifier(TwoCommunityClassifier):
    def __init__(self,graph,B=None,category=None,Q=None):
        super().__init__(graph,B)
        if category:
            self.category=category
        if Q:
            self.Q=Tree(Q)
            self.Q.left
    
    def Beq(self,nodes):
        #compute the equivalent matrix Beq

        Beq=self.B[nodes,:][:,nodes]
        Beq-=np.diagonal(np.sum(Beq,axis=1))
        return Beq

    def fit(self,graph=None,B=None,category=None,Q=None,direction=None):
    
        if graph is None:
            # clf=TwoCommunityClassifier(self.G,self.B)
            graph=self.G
        if category:
            self.category=category
        
        clf=TwoCommunityClassifier(graph,B,self.category)
        clf.fit()
        if Q is None:
            self.Q=Tree(clf.Q)
        # else:
        #     if direction="Left":
        #         self.Q.left=Tree(clf.Q)
        #     else:
        #         self.Q.right=Tree(clf.Q)

        if clf.done:
            # If undivisible graph, terminate the classification
            for i,node in enumerate(graph.nodes):
                self.category[node].append(1 if clf.leading_eigenvector[i]>=0 else -1)
        else:
            #If the graph is divisible, divide it and apply the algorithm to the subgraphs.
            nodes=np.arange(len(clf.leading_eigenvector))
            nodes_positive=nodes[clf.leading_eigenvector>=0]
            nodes_negative=nodes[clf.leading_eigenvector<0]
            
            subgraph_positive=graph.subgraph(nodes_positive)
            subgraph_negative=graph.subgraph(nodes_negative)

            B_positive=self.Beq(nodes_positive)
            B_negative=self.Beq(nodes_negative)
            
            self.fit(subgraph_positive,B_positive,category)
            self.fit(subgraph_negative,B_negative,category)
            # Q_positive=np.einsum("i,ij,j",s,self.B_positive,s)/(4*self.m)
            # Q_negative=np.einsum("i,ij,j",s,self.B_negative,s)/(4*self.m)
            # Push the DQ values where relevant
    
    # def modularity(self,Q):
        # pass
        


# ---------------------------------- Results --------------------------------- #

# clf=TwoCommunityClassifier(G)
# clf.fit()

# # print("Leading eigenvector")
# # print(clf.leading_eigenvector)
# print("Communauté")
# print(clf.category)
# print("Q: %f"%(clf.Q))

clfN=NCommunityClassifier(G)  
clfN.fit()  