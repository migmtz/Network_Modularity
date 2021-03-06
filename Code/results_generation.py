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
from NCommunities import NCommunitiesClassifier
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1) 

# import ipdb; ipdb.set_trace()

# ---------------------------------------------------------------------------- #
#                               Graph data import                              #
# ---------------------------------------------------------------------------- #

# More examples to add

G_List=[]
G_List.append(networkx.generators.karate_club_graph())
g_polbooks=networkx.read_gml('./data/polbooks.gml')
g_polbooks.name="Books politics" #set names to graphs
G_List.append(g_polbooks)


# ---------------------------------------------------------------------------- #
#                                Modularity plot                               #
# ---------------------------------------------------------------------------- #

#The purpose of the section is to plot modularity as a function of the number of communities for all four algorithms

# --------------------------------- Settings --------------------------------- #

G=networkx.read_gml('./data/polbooks.gml') #find a larger network (does not exceed 4 communities with this one)
n=10

# ----------------------------- Plot construction ---------------------------- #

# import ipdb; ipdb.set_trace()
clfN=NCommunitiesClassifier(G,Newman2CommunityClassifier,n)
clfN.fit()
# print(clfN.Q_History)
plt.plot(np.arange(1,n+1),clfN.padded_modularity_sequence(n),label="N06")
clfN=NCommunitiesClassifier(G,DA2communityClassifier,n)
clfN.fit()
# print(clfN.Q_History)
plt.plot(np.arange(1,n+1),clfN.padded_modularity_sequence(n),label="DA")
clfN=NCommunitiesClassifier(G,GN2communityClassifier,n)
clfN.fit()
# print(clfN.Q_History)
plt.plot(np.arange(1,n+1),clfN.padded_modularity_sequence(n),label="GN")
clfN=NCommunitiesClassifier(G,SP2CcommunityClassifier,n)
clfN.fit()
# print(clfN.Q_History)s
plt.plot(np.arange(1,n+1),clfN.padded_modularity_sequence(n),label="Spectral")
plt.xlabel("Number of communities")
plt.ylabel("Modularity")
plt.legend()
plt.show()

# ---------------------------------------------------------------------------- #
#                           Table optimal modularity                           #
# ---------------------------------------------------------------------------- #

#take the maximal over a sequence of n modularity

for g in G_List:
    clfN=NCommunitiesClassifier(g,DA2communityClassifier,n)
    clfN.fit()
    print("Maximal modularity and corresponding number of communities for graph %s, classifier DA: Q=%.3f, N=%d"%(g.name,clfN.return_optimal()[0],clfN.return_optimal()[1]))
    clfN=NCommunitiesClassifier(g,Newman2CommunityClassifier,n)
    clfN.fit()
    print("Maximal modularity and corresponding number of communities for graph %s, classifier N06: Q=%.3f, N=%d"%(g.name,clfN.return_optimal()[0],clfN.return_optimal()[1]))
    clfN=NCommunitiesClassifier(g,GN2communityClassifier,n)
    clfN.fit()
    print("Maximal modularity and corresponding number of communities for graph %s, classifier GN: Q=%.3f, N=%d"%(g.name,clfN.return_optimal()[0],clfN.return_optimal()[1]))
    clfN=NCommunitiesClassifier(g,SP2CcommunityClassifier,n)
    clfN.fit()
    print("Maximal modularity and corresponding number of communities for graph %s, classifier Spectral: Q=%.3f, N=%d"%(g.name,clfN.return_optimal()[0],clfN.return_optimal()[1]))


# ---------------------------------------------------------------------------- #
#                                Plot of a graph                               #
# ---------------------------------------------------------------------------- #

# ------------------------------ Two communities ----------------------------- #

G=networkx.generators.karate_club_graph()
clf2=Newman2CommunityClassifier(G)
clf2.fit()
plot_communities(G,clf2)
plot_communities_eigen(G,clf2)

# ------------------------------- N communities ------------------------------ #

G=networkx.read_gml('./data/polbooks.gml')
clfN=NCommunitiesClassifier(G,Newman2CommunityClassifier)
clfN.fit()
plot_communities(G,clfN)
# plot_communities_eigen(G,clfN)