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
from networkx.algorithms import community


# np.random.seed(1)
# G = networkx.read_gml('./data/netscience.gml') #1589 nodes
# G = networkx.read_gml('./data/polbooks.gml') #105 nodes
# G=networkx.davis_southern_women_graph() #32 nodes
# G=networkx.florentine_families_graph() #15 nodes

# import ipdb; ipdb.set_trace()
# print(len(G))

class NCommunitiesClassifier():

    def __init__(self,G,BinaryClassifier,N=None,category=None):
        self.G=G
        self.m=np.sum(to_numpy_matrix(G))/2
        self.BinaryClassifier=BinaryClassifier
        self.community_count=1
        self.optimal_stop=False
        self.N=N
        if N is None:
            self.optimal_stop=True #stop algorithm using the natural criterion.
        if category is None:
            self.category={node:[] for node in G.nodes}
        else:
            self.category=category
        self.Q=0
        self.Q_History=[0]
        self.optimal=(self.Q,self.community_count)
        self.communities={}
        self.done=False

    def compute_communities(self,category=None):
        # import ipdb; ipdb.set_trace()
        if category is None:
            category=self.category
        self.communities={}
        # import ipdb; ipdb.set_trace()
        for k,v in category.items():
            if str(v) not in self.communities:
                self.communities[str(v)]=[k]
            else:
                self.communities[str(v)].append(k)
        self.communities=list(self.communities.values())
        
    def compute_modularity(self):
        # m=self.m #same m at each recursion step? seems strange
        # B=clf.A-np.dot(clf.k,clf.k.transpose())/(2*m)
        # if (clf.G!=self.G):
        #     #Adapt formula for subgraphs
        #     B-=np.diagonal(np.sum(B,axis=1))
        # Q=np.einsum("i,ij,j",clf.s,B,clf.s)/(4*m)
        # import ipdb; ipdb.set_trace()
        self.compute_communities()
        Q=community.modularity(self.G,self.communities)
        return Q
    
    def padded_modularity_sequence(self,n):
        pad=[0 for i in range(max(n-len(self.Q_History),0))]
        self.Q_History+=pad
        return self.Q_History

    def fit(self,G=None,verbose=False):
        # import ipdb; ipdb.set_trace()
        try:
            if G is None:
                G=self.G
            clf=self.BinaryClassifier(G)
            clf.fit()
            # DQ=self.compute_modularity(clf)
            # import ipdb; ipdb.set_trace()
            if self.optimal_stop:
                if clf.Q<0:
                    self.done=True
            if not self.done:
                for node in clf.category:
                    self.category[node]+=clf.category[node]
                import ipdb; ipdb.set_trace()
                # self.Q=max(0,self.Q+DQ) #consider only positive values
                self.Q=self.compute_modularity()
                self.Q_History.append(self.Q)
                self.community_count+=1
                if self.N is not None and self.N==self.community_count:
                    self.done=True
                if self.Q>self.optimal[0]:
                    self.optimal=(self.Q,self.community_count)
                clfN_positive=NCommunitiesClassifier(clf.G_positive,self.BinaryClassifier,category=self.category)
                clfN_positive.fit()
                clfN_negative=NCommunitiesClassifier(clf.G_negative,self.BinaryClassifier,category=self.category)
                clfN_negative.fit()
        except:
            if verbose:
                print("Error while running the NCommunities")  

G = networkx.read_gml('./data/polbooks.gml') #105 nodes

clfN=NCommunitiesClassifier(G,Newman2CommunityClassifier)
clfN.fit()
print(clfN.Q_History)
import ipdb; ipdb.set_trace()
# import ipdb; ipdb.set_trace()
# clfN.compute_communities()

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

