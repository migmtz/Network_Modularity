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
import copy



class NCommunitiesClassifier():

    def __init__(self,G,BinaryClassifier,N=None,category=None,G0=None,Q_History=None):
        self.G=G
        self.m=np.sum(to_numpy_matrix(G))/2
        self.BinaryClassifier=BinaryClassifier
        self.N=N
        self.Q=0
        self.done=False
        self.optimal_stop=False
        if N is None:
            self.optimal_stop=True #stop algorithm using the natural criterion.
        if category is None:
            self.category={node:[] for node in G.nodes}
        else:
            self.category=category
        if Q_History is None:
            self.Q_History=[0]
        else:
            self.Q_History=Q_History
        if G0 is None:
            self.G0=G
        else:
            self.G0=G0

    def compute_communities(self,category):
        """
            Convert category to community
        """
        self.communities={}
        for k,v in category.items():
            if str(v) not in self.communities:
                self.communities[str(v)]=[k]
            else:
                self.communities[str(v)].append(k)
        self.communities=list(self.communities.values())
        
    def compute_modularity(self,G0=None,category=None):
        # m=self.m #same m at each recursion step? seems strange
        # B=clf.A-np.dot(clf.k,clf.k.transpose())/(2*m)
        # if (clf.G!=self.G):
        #     #Adapt formula for subgraphs
        #     B-=np.diagonal(np.sum(B,axis=1))
        # Q=np.einsum("i,ij,j",clf.s,B,clf.s)/(4*m)
        # import ipdb; ipdb.set_trace()
        if G0 is None:
            G0=self.G0
        if category is None:
            category=self.category
        self.compute_communities(category)
        Q=community.modularity(self.G0,self.communities)
        return Q

    def return_optimal(self):
        Q=np.max(self.Q_History)
        N=np.argmax(self.Q_History)
        return Q,N
    
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
            if self.optimal_stop:
                #Compute modularity before-hand and see it decreases overall
                category=copy.deepcopy(self.category)
                for node in clf.category:
                    category[node]+=clf.category[node]
                Q_temp=self.compute_modularity(self.G0,category)
                if Q_temp<self.Q_History[-1]: #if there is a decrease in Q, interrupt the process.
                    self.done=True
            else:
                if len(self.Q_History)==self.N:
                    self.done=True
            if not self.done:
                for node in clf.category:
                    self.category[node]+=clf.category[node]
                self.Q=self.compute_modularity(self.G0,self.category)
                self.Q_History.append(self.Q)        
                clfN_positive=NCommunitiesClassifier(clf.G_positive,self.BinaryClassifier,N=self.N,category=self.category,G0=self.G0,Q_History=self.Q_History)
                clfN_positive.fit()
                clfN_negative=NCommunitiesClassifier(clf.G_negative,self.BinaryClassifier,N=self.N,category=self.category,G0=self.G0,Q_History=self.Q_History)
                clfN_negative.fit()
        except:
            if verbose:
                print("Error while running the NCommunities")  
            return None

# G=networkx.read_gml('./data/polbooks.gml') #find a larger network (does not exceed 4 communities with this one)
# clfN=NCommunitiesClassifier(G,DA2communityClassifier)
# clfN.fit()
# print(clfN.Q_History)


# G=networkx.read_gml('./data/polbooks.gml') #find a larger network (does not exceed 4 communities with this one)
# clfN=NCommunitiesClassifier(G,DA2communityClassifier,10)
# clfN.fit()
# print(clfN.Q_History)


#Weird: they don t follow the same sequence, even at first! 
# import ipdb; ipdb.set_trace()