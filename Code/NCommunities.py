import networkx
from networkx.convert_matrix import to_numpy_matrix
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from matplotlib.pyplot import cm
from plot_generators import *
from Newman06 import Newman2CommunityClassifier
from DuchArenas import DA2communityClassifier
from Girvan_Newman import GN2communityClassifier
from Spectral_Clustering import SP2CcommunityClassifier

# np.random.seed(1)
# G = networkx.read_gml('./data/netscience.gml') #1589 nodes
# G = networkx.read_gml('./data/polbooks.gml') #105 nodes
# G=networkx.davis_southern_women_graph() #32 nodes
# G=networkx.florentine_families_graph() #15 nodes

# import ipdb; ipdb.set_trace()
# print(len(G))

class NCommunitiesClassifier():

    def __init__(self,G,BinaryClassifier,N=None):
        self.G=G
        self.m=np.sum(to_numpy_matrix(G))/2
        self.BinaryClassifier=BinaryClassifier
        self.community_count=1
        self.optimal_stop=False
        self.N=N
        if N is None:
            self.optimal_stop=True #stop algorithm using the natural criterion.
        self.category={node:[] for node in G.nodes}
        self.Q=0
        self.Q_History=[0]
        self.optimal=(self.Q,self.community_count)
        self.done=False

    def compute_modularity(self,clf):
        m=self.m #same m at each recursion step? seems strange
        B=clf.A-np.dot(clf.k,clf.k.transpose())/(2*m)
        if (clf.G!=self.G):
            #Adapt formula for subgraphs
            B-=np.diagonal(np.sum(B,axis=1))
        Q=np.einsum("i,ij,j",clf.s,B,clf.s)/(4*m)
        return Q
    
    def padded_modularity_sequence(self,n):
        pad=[0 for i in range(max(n-len(self.Q_History),0))]
        self.Q_History+=pad
        return self.Q_History

    def fit(self,G=None,verbose=False):
        try:
            if G is None:
                G=self.G
            clf=self.BinaryClassifier(G)
            clf.fit()
            DQ=clf.Q #self.compute_modularity(clf)
            # import ipdb; ipdb.set_trace()
            if self.optimal_stop:
                if DQ<0:
                    self.done=True
            if not self.done:
                for node in clf.category:
                    self.category[node]+=clf.category[node]
                self.Q=max(0,self.Q+DQ) #consider only positive values
                self.Q_History.append(self.Q)
                self.community_count+=1
                if self.N is not None and self.N==self.community_count:
                    self.done=True
                if self.Q>self.optimal[0]:
                    self.optimal=(self.Q,self.community_count)
                self.fit(clf.G_positive)
                self.fit(clf.G_negative)
        except:
            if verbose:
                print("Error while running the NCommunities")  

# G = networkx.read_gml('./data/polbooks.gml') #105 nodes

# clfN=NCommunitiesClassifier(G,GN2communityClassifier)
# clfN.fit()
# print(clfN.Q)
# print(clfN.community_count)
# print(clfN.Q_History)
# print(clfN.padded_modularity_sequence(10))

# clf2=GN2communityClassifier(G)
# clf2.fit()
# clf2=GN2communityClassifier(clf2.G_positive)
# clf2.fit()
# Beq=clf2.B-np.diag(np.sum(clf2.B,axis=1))
# Q=np.einsum("i,ij,j",clf2.s,Beq,clf2.s)/(4*clf2.m)
# print(Q)

