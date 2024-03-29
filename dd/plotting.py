import os
import time

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import networkx as nx
import pandas as pd

def plot_macro_gc(edge_weights, node_weights, trials):

    for i in range(trials):            # the range is over the amount of simulations performed.
        subset = pd.DataFrame(edge_weights[:, edge_weights[i]:edge_weights[i+1]])
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(24, 8))
        gs = GridSpec(nrows=1, ncols=3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Pairwise Granger-causality Matrix", fontsize=30, fontweight='bold', pad=16)
        sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, annot=True)
        ax0.invert_yaxis()

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title("GC-graph of an coupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("2-Macro on GC-graph of coupled {0}-node SJ3D model".format(int(len(subset))), fontsize=30, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=node_weights[:, i], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig





def plot_dd_simulations(dynamical_dependence_values, macrosize):

    edge_cmap = sns.color_palette("YlOrBr", as_cmap=True)

    coupling = [str(round(float(x), 2)) for x in 10**np.r_[-2:-0.7:20j]]
    noise = [str(round(float(x), 3)) for x in 10**np.r_[-6:-1:20j]]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(nrows=1, ncols=1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title('Values of DD of {0}-macro for each parameter regime: \n (noise, coupling) across the parameter sweep'.format(macrosize), fontsize=14, fontweight='bold', pad=16)
    ax0 = sns.heatmap(dynamical_dependence_values, cmap=edge_cmap, cbar_kws={'label': 'Dynamical Dependence Value'}, linecolor='black', linewidths=.6)
    # ax0.set_xscale('log')
    # ax0.set_yscale('log')
    ax0.set_xlabel('Noise', fontsize=12, fontweight='bold', labelpad=10)
    ax0.set_ylabel('Global Coupling', fontsize=12, fontweight='bold', labelpad=10)

    ax0.set_xticklabels(noise, fontsize=10, rotation=45)
    ax0.set_yticklabels(coupling, fontsize=10, rotation=45)
    ax0.invert_yaxis()
    ax0.figure.axes[-1].yaxis.label.set_size(12)

    return fig


def plot_gc(edge_weights):

    # the range is over the amount of simulations performed.
    subset = pd.DataFrame(edge_weights)
    # subset.columns = ['0', '1', '2']
    # subset.index = ['0','1','2']
    G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples
    
    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows
    # PLOTTING THE PWCGC MATRIX and GRAPH.
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=width_ratios, height_ratios=height_ratios)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Pairwise conditional G-causal Matrix", fontsize=18, fontweight='bold', pad=16)
    sns.heatmap(subset, cmap=mpl.cm.bone_r, center=0.5, linewidths=.6, linecolor='black',annot=True, cbar_kws={'label': 'G-causal estimate values'})

    
    ax0.invert_yaxis()


    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("PWC G-causal graph of {0}-node MVARMA model".format(int(len(subset))), fontsize=18, fontweight='bold', pad=16)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color='lightgray', linewidths=1.0, edgecolors='black')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights,node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
    edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])


def plot_nweights(eweights, nweights, macrosize, opt_number):
        
        subset = pd.DataFrame(eweights)
        # subset.columns = ['0', '1', '2']
        # subset.index = ['0','1','2']
        G = nx.from_pandas_adjacency(subset, create_using=nx.MultiDiGraph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))                    # remove self edges
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) # Extracts the edges and their corresponding weights into different tuples

        # PLOTTING THE PWCGC MATRIX, GRAPH AND MACRO PROJECTION ON GRAPH.
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(nrows=1, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("{0}-Macro on GC-graph of coupled {1}-node model".format(int(macrosize), int(len(subset))), fontsize=18, fontweight='bold', pad=16)
        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=1600, node_color=nweights[:,opt_number], cmap=plt.cm.Blues, linewidths=1.0, edgecolors='black') # nweights[:,0] will plot the optimal projection of the first macro variable on the graph.
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10.0, edgelist=edges, edge_color=weights, node_size=1600, width=3.0, connectionstyle='arc3,rad=0.13', edge_cmap=mpl.cm.bone_r)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="helvetica")
        edge_labels = dict([((u, v,), f"{d['weight']:.1f}") for u, v, d in G.edges(data=True)])

        return fig


def plot_optp(preopthist, preoptdist):

    n_cols = 2
    n_rows = 1
    # Calculate the width and height ratios
    width_ratios = [1.2] * n_cols
    height_ratios = [1] * n_rows


    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(nrows=n_rows, ncols=n_cols, width_ratios=
                  width_ratios, height_ratios=height_ratios)
    # Set the width and height ratios for the GridSpec
    #gs.update(width_ratios=width_ratios, height_ratios=height_ratios)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Pre-optimisation History', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    for i in range(len(preopthist)):
        ax1 = sns.lineplot(data=preopthist[i][0][:, 0], legend=False, dashes=False, palette='bone_r', linewidth=0.6)

    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Local-Optima Distances', fontweight='bold', fontsize=14)
    ax2 = sns.heatmap(preoptdist['goptp'], cmap=cmap, center=np.max(preoptdist['goptp'])/2, cbar_kws={'label': 'Orthogonality of subspaces'})
    ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax2.invert_yaxis()

    return fig


def plot_opto(opthist, optdist):

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=1, ncols=2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Optimisation History: ', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    ax1.set_xscale('log')
    for i in range(len(opthist)):
        ax1 = sns.lineplot(data=opthist[i][0][:, 0], legend=False, dashes=False, palette='bone_r')

    cmap = sns.color_palette("bone_r", as_cmap=True)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Local-Optima Distances: ', fontweight='bold', fontsize=18)
    ax2 = sns.heatmap(optdist['gopto'], cmap=cmap, center=np.max(optdist['gopto'])/2, cbar_kws={'label': 'Othogonality of subspaces'})
    ax2.set_xlabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Optimisation run', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 8)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), fontsize = 8)
    ax2.invert_yaxis()

    return fig

def plot_opthist(opthist):

    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(nrows=1, ncols=1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Optimisation History: ', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Dynamical Dependence', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    ax1.set_xscale('log')
    for i in range(len(opthist)):
        ax1 = sns.histplot(data=opthist[i][0][:, 0], legend=False, dashes=False, palette='bone_r')

    return fig

