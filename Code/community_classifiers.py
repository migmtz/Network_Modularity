import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
from scipy.sparse import csc_matrix

# --------------------------- Generate sample graph -------------------------- #

G=networkx.generators.atlas.graph_atlas(100)

print("Matrice d'adjacence")
print(to_numpy_matrix(G))
print("Vecteurs des degrés")
print(np.sum(to_numpy_matrix(G),axis=1))

# ----------------------- 2-Communities classifier ---------------------- #

class TwoCommunityClassifier():
    
    def __init__(self,graph):
        self.G=graph
        self.A=to_numpy_matrix(graph)
        self.k=np.sum(self.A,axis=1)
        self.m=np.sum(self.k)
        self.B=self.A-np.dot(self.k,self.k.transpose())/self.m
        self.leading_eigenvector=None
        self.category=None
        self.indivisible=False
    
    def fit(self,eps=0.5,max_iteration=1000):
        # Compute the simple power method (TBC)
        u=np.ones(self.A.shape[0])/self.A.shape[0]
        self.k=np.ravel(self.k)
        A=csc_matrix(self.A)
        delta=2*eps
        step=0
        while delta>eps and step<max_iteration:
            up=A.dot(u)-np.dot(u,self.k)*self.k/self.m
            up/=np.linalg.norm(up)
            step+=1
            delta=np.linalg.norm(up-u)
            # does not converge for eps =0.1 (suspicious)
        self.leading_eigenvector=up

        #Alternatively, use built in function for eigenvalues
        vals,vecs=eig(self.B)
        self.leading_eigenvector=vecs[:,np.argmax(vals)]
        if np.max(self.leading_eigenvector)*np.min(self.leading_eigenvector)>=0:
            self.indivisible=True
        self.category={node: 1 if self.leading_eigenvector[i]>=0 else -1  for i,node in enumerate(self.G.nodes)}
        # np.ravel((self.leading_eigenvector>=0).astype(int))
        # self.category[self.category==0]=-1

    def Q(self):
        s=list(self.category.values())
        return np.einsum("i,ij,j",s,self.B,s)/(4*self.m)

# ------------------------- N-Communities classifier ------------------------- #

# The idea is to apply the previous algorithm recursively each community until convergence of the process.
class NCommunityClassifier(TwoCommunityClassifier):
    def __init__(self,graph):
        super().__init__(graph)
    
    def fit(self,subgraph):
        #Apply the Two Communities classification recursively until the communities cannot be divided themselves
        #Careful in the updates of the values
        if subgraph==None:
            # first call of the recursive stack 
            clf=TwoCommunityClassifier(self.graph)
            clf.fit()
            if clf.indivisible:
                # No need to proceed further
                return clf.category
            else:
                # self.fit(sub1,sub2)
                # Apply the recursive step
        else:
            #all subgraphs calls
            clf=TwoCommunityClassifier(self.graph)
            clf.fit()
            if clf.indivisible:
                # No need to proceed further
                return clf.category
            else:

                # self.fit(sub1)


# ---------------------------------- Results --------------------------------- #

clf=TwoCommunityClassifier(G)
clf.fit()

# print("Leading eigenvector")
# print(clf.leading_eigenvector)
print("Communauté")
print(clf.category)
print("Q: %f"%(clf.Q()))

clfN=NCommunityClassifier(G)    