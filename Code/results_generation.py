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

np.random.seed(1) 

# ---------------------------------------------------------------------------- #
#                               Graph data import                              #
# ---------------------------------------------------------------------------- #

G = networkx.read_gml('./data/polbooks.gml')


clfN=NCommunitiesClassifier(G,GN2communityClassifier,2)
clfN.fit()

print("Modularity %.3f"%(clfN.Q))