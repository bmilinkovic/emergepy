from itertools import combinations
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec
import os
import scipy.io as sio
import matplotlib as mpl

# If you want to read CSV file into a NumPy array
array_data = np.genfromtxt('dist_8.csv', delimiter=',')

# If you want to read CSV file into a Pandas DataFrame
#df = pd.read_csv('dist.csv')

n = 9  # Replace with your desired value of n
m = 8  # Replace with your desired value of m

def plot_subspace_emergence(Loptx, n, m):
    """ 
    Plots the distance between the optimal macroscopic coarse-graining Loptx 
    and the optimal microscopic subset of the original system, giving us an 
    optimal subset of the original system that is emergent.
    

    Args:
        Loptx (numpy array): the distance between the optimal macroscopic coarse-graining and all microscopic subset of the original system.
        n (int): the number of nodes in the original system.
        m (int): the number of nodes in the optimal macroscopic coarse-graining.

    Returns:
        None

    """

    c = list(combinations(range(1, n + 1), m)) # all n choose m combinations of the original system
    nc = len(c)

    for k in range(nc): 
        if Loptx[k] > 0.1:
            print(f"{k + 1:3d} : {' '.join(str(num) for num in c[k])} : {Loptx[k]:6.4f}")

    # Now we will plot all the set of n choose k subspaces on the x-axis and the value in Loptx on the y-axis as a bar graph
    


    plt.figure(figsize=(13, 5))
    plt.bar(range(nc), Loptx, color="blue")
    plt.xlabel("Subspace", fontsize=16, fontweight="bold")
    plt.ylabel(f"1 - Normalised subspace distance", fontsize=14, fontweight="bold")
    # the following line sets the title as the distance between the optimal {0}-macro coarse-graining and all {0}-subsets subset of the original {1}-node system, where the {0} will be replaced by the value of m and {1} will be replaced by the value of n
    plt.title(f"Distance b/w {m}-macro and {m}-subsets of the {n}-node system", fontsize=18, fontweight="bold")
    # the following line will set all xticks to 'empty' if the distance is less than 0.1
    plt.xticks(range(nc), ['' if Loptx[k] < 0.1 else f"{k + 1:3d} : {' '.join(str(num) for num in c[k])}" for k in range(nc)], rotation=90, fontsize=10)
    # in the following line we will highlight the bar indicating the emergent subspace with the highest distance
    plt.bar(np.argmax(Loptx), Loptx[np.argmax(Loptx)], color="red")
    # the following line gets rid of the top frame line and the right frame line
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    # the following line saves the plot as a EPS file in the same directory as this script and saves it with the corresponding number from the input filename
    plt.savefig(f"dist_plot_{m}.eps")
    # the following line saves the plot as a PNG file in the same directory as this script and saves it with the corresponding number from the input filename
    plt.savefig(f"dist_plot_{m}.png")
    plt.show()


plot_subspace_emergence(array_data, n, m)



optimisation_dir = '/Users/borjanmilinkovic/Documents/gitdir/ssdi/networks/models/'
nweight_dir = '/Users/borjanmilinkovic/Documents/gitdir/TVBEmergence/networks/models/'

# load in data
eweight = sio.loadmat(os.path.join(optimisation_dir, 'sim_model_0339_15_06_2023.mat'))      # load in edge weights
eweight = eweight['eweight']    # extract edge weights

# laoad in node weights and plot that

nweight = sio.loadmat(os.path.join(nweight_dir, 'nweight_mdim_8.mat'))      # load in node weights
nweight = nweight['nweight']    # extract node weights



def plot_nweights(eweights, nweights, macrosize, opt_number):
        
        subset = pd.DataFrame(eweights)

        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        # Define the mapping from old labels to new labels
        mapping = {node: node + 1 for node in G.nodes}

        # Relabel the nodes in the graph
        G = nx.relabel_nodes(G, mapping)

        # Plot the graph with the new labels
        #nx.draw(G, with_labels=True)
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(nrows=1, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        
        #ax0.set_title("{0}-Macro on GC-graph of coupled {1}-node model".format(int(macrosize), int(len(subset))), fontsize=18, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])
        # show the figure
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('macro_{0}_on_GC_graph_{1}_node_model.png'.format(int(macrosize), int(len(subset))))
        plt.savefig('macro_{0}_on_GC_graph_{1}_node_model.eps'.format(int(macrosize), int(len(subset))))
        plt.show()



fig = plot_nweights(eweight, nweight, 8, 0)
