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

# G = networkx.read_gml('./data/netscience.gml') #1589 nodes
G = networkx.read_gml('./data/polbooks.gml') #105 nodes
# G=networkx.davis_southern_women_graph() #32 nodes
# G=networkx.florentine_families_graph() #15 nodes

# import ipdb; ipdb.set_trace()
# print(len(G))

class NCommunitiesClassifier():

    def __init__(self,G,BinaryClassifier,N=None):
        self.G=G
        self.m=np.sum(to_numpy_matrix(G))/2
        self.BinaryClassifier=BinaryClassifier
        self.commmunity_count=1
        self.optimal_stop=False
        self.N=N
        if N is None:
            self.optimal_stop=True #stop algorithm using the natural criterion.
        self.category={node:[] for node in G.nodes}
        self.Q=0
        self.Q_History=[0]
        self.optimal=(self.Q,self.commmunity_count)
        self.done=False

    def compute_modularity(self,G,s):
        m=self.m #same m at each recursion step? seems strange
        A=to_numpy_matrix(G)
        k=np.sum(to_numpy_matrix(G),axis=1)
        B=A-np.dot(k,k.transpose())/(2*m)
        if (G!=self.G):
            #Adapt formula for subgraphs
            B-=np.diagonal(np.sum(B,axis=1))
        Q=np.einsum("i,ij,j",s,B,s)/(4*m)
        return Q

    def fit(self,G=None,verbose=False):
        try:
            if G is None:
                G=self.G
            clf=self.BinaryClassifier(G)
            clf.fit()
            DQ=self.compute_modularity(G,clf.s)
            if self.optimal_stop:
                if DQ<0:
                    self.done=True
            if not self.done:
                for node in clf.category:
                    self.category[node]+=clf.category[node]
                self.Q+=DQ
                self.Q_History.append(self.Q)
                self.commmunity_count+=1
                if self.N is not None and self.N==self.commmunity_count:
                    self.done=True
                if self.Q>self.optimal[0]:
                    self.optimal=(self.Q,self.commmunity_count)
                self.fit(clf.G_positive)
                self.fit(clf.G_negative)
        except:
            if verbose:
                print("Error while running the NCommunities")  
            self.Q=0
            self.Q_History.append(self.Q)

# clfN=NCommunitiesClassifier(G,DA2communityClassifier)
# clfN.fit()
# print("Optimal settings for graph %s, classifier DA: Q=%.3f, N=%d"%(G.name,clfN.Q,clfN.commmunity_count))

# clfN=NCommunitiesClassifier(G,Newman2CommunityClassifier)
# clfN.fit()
# print("Optimal settings for graph %s, classifier N06: Q=%.3f, N=%d"%(G.name,clfN.Q,clfN.commmunity_count))


# clfN=NCommunitiesClassifier(G,GN2communityClassifier)
# clfN.fit()
# print("Optimal settings for graph %s, classifier GN: Q=%.3f, N=%d"%(G.name,clfN.Q,clfN.commmunity_count))

# clfN=NCommunitiesClassifier(G,SP2CcommunityClassifier)
# clfN.fit()
# print("Optimal settings for graph %s, classifier Spectral: Q=%.3f, N=%d"%(G.name,clfN.Q,clfN.commmunity_count))
