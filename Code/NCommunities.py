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

G = networkx.read_gml('./data/polbooks.gml')
# G=networkx.generators.karate_club_graph()


class NCommunitiesClassifier():

    def __init__(self,G,BinaryClassifier,N=None):
        self.G=G
        self.m=np.sum(to_numpy_matrix(G))/2
        self.BinaryClassifier=BinaryClassifier
        self.commmunity_count=1
        if N is None:
            self.N=10
        else:
            self.N=N
        self.category={node:[] for node in G.nodes}
        self.Q=0
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

    def fit(self,G=None):
        if G is None:
            G=self.G
        clf=self.BinaryClassifier(G)
        clf.fit()
        DQ=self.compute_modularity(G,clf.s)
        if not self.done:
        # if not self.done:
            #interrupt calculation when there is no more progress to be done.
            #also possible to steer computation with the required number of communities
            for node in clf.category:
                self.category[node]+=clf.category[node]
            self.Q+=DQ
            self.commmunity_count+=1
            if self.N is not None and self.N==self.commmunity_count:
                self.done=True
            # import ipdb; ipdb.set_trace()
            self.fit(clf.G_positive)
            self.fit(clf.G_negative)
