#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:08:27 2020

@author: hh1915
"""
import osmnx as ox
import networkx as nx

import numpy as np
from numpy import save, load
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle
from math import radians, cos, sin, asin, sqrt 
from TSP_Solver import TSP


class Sub_Graph:
    def __init__(self, D_sub, S_sub, S_full, A_sub, A_full, n_sub, nodes_pos_sub ,nodes_pos):
        self.D_sub = D_sub
        self.S_sub = S_sub
        self.S_full = S_full     
        self.A_sub = A_sub
        self.A_full = A_full     
        self.N_nodes = n_sub
        self.Pos_nodes_sub = nodes_pos_sub
        self.Pos_nodes_full = nodes_pos

def A_meters(nodes_pos):
    lat_min = np.min(nodes_pos[:,0])
    lat_max = np.max(nodes_pos[:,0])
    
    lon_min = np.min(nodes_pos[:,1])
    lon_max = np.max(nodes_pos[:,1])
    
    dist_lat = lat_max - lat_min
    dist_lon = lon_max - lon_min

    
    A = dist_lat * dist_lon 
    
    return A


def Gen_road_network(G):
    """
    Generate Full Road Network as a Graph
    
    """
    
    # Sample graph


    # Size
    NoNodes = G.number_of_nodes()
    NoEdges = G.number_of_edges() 
    
    # Adjacency Matrix
    A = nx.adjacency_matrix(G).todense() 
    
    # Nodes Dict
    nodes = dict(G.nodes())
    nodes_idx = list(nodes.keys())
    nodes_pos = np.zeros((NoNodes,2))
    for i, node_idx in enumerate(nodes_idx):
        nodes_pos[i,:] = [ nodes[node_idx]['y'],nodes[node_idx]['x'] ] # longtitude, latitude
    

        
    """
    OUTPUT ONE: Distance Matrix by Dijkstra algorithm
    
    """
    
    
    print('computing D by Dijkstra algorithm...')
    D_dict = dict(nx.all_pairs_dijkstra_path_length(G,weight='length')) # shortest path between a pair of nodes with weighted edges
    
    
    D_np = np.zeros((NoNodes,NoNodes))
    for i, node_str in enumerate(nodes_idx):
        for j, node_end in enumerate(nodes_idx):
            D_np[i,j] = D_dict[node_str][node_end]
    
    
    """
    OUTPUT TWO: Covariance Matrix
    
    """
    print('computing S...')
    mpd = 111111
    nodes_pos = nodes_pos * mpd # degree to meters
   
    n = np.shape(nodes_pos)[0]
    X_mean = np.mean(nodes_pos, axis=0)
    A = nodes_pos - X_mean
    S_np = 1/n*np.matmul(A.T,A)
    
    print(A)
        
    return D_np, S_np, NoNodes, nodes_pos



def Gen_subgraphs(D_full, S_full, A_full, n_full, nodes_pos, Size_lst, N_Per_Size ):

    graphs = []
    for i , Num in enumerate(Size_lst):
        for j in range(N_Per_Size):
            nodes_subgraph = np.random.choice(n_full, Num, replace=False)
            nodes_pos_sub = nodes_pos[nodes_subgraph,:]
            
            #
            D_subgraph = D_full[nodes_subgraph,:][:,nodes_subgraph]
            
            #
            X = nodes_pos_sub
            n_sub = np.shape(X)[0]
            X_mean = np.mean(X, axis=0)
            A = X - X_mean
            S_subgraph = 1/n_sub*np.matmul(A.T,A)
            
            #
            S_fullgraph = S_full
            
            #
            A_subgraph = A_meters(nodes_pos_sub)
            
            #
            A_fullgraph = A_full
            
            #
            n_subgraph = Num
            
            subgraph = Sub_Graph(D_subgraph,
                                 S_subgraph,
                                 S_fullgraph,
                                 A_subgraph,
                                 A_fullgraph,
                                 n_subgraph,
                                 nodes_pos_sub,
                                 nodes_pos)
            
            graphs.append(subgraph)          
    return graphs



########################################################################################################

D_full = load('Data/D_full_PEK_5R.npy', allow_pickle = True)
n = np.shape(D_full)[0] 
closeness_centrality = np.sum(D_full, axis=0)/n
mean_dist = np.sum(np.sum(D_full, axis=0))/(n**2)

print('mean distance:',mean_dist)


"""
Generate Large real-world graph

"""
Gen = False
if Gen == True:
    # Enforce strong connectivity for the directed graph
    # Selected the strongly connected giant component
    print('generating graph...')
    # NYC
    #G = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')
    
    # LDN
    #G = ox.graph_from_address('London, uk', dist=7700, network_type='drive', simplify=True)
    
    # PEK
    #G = ox.graph_from_point((39.950695, 116.433546), dist = 7700/2, network_type='drive', simplify=True)
    #G = ox.graph_from_place('Beijing, China', network_type='drive', simplify=True)
    
    G = ox.graph_from_point((39.920378, 116.396591), dist = 7700*2.5, dist_type='network', network_type='drive', simplify=True)

    print('computing the largest component...')
    Gcc = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    
    
    # compute closeness centrality
    print('computing closeness centrality...')
    #closeness_centrality = nx.closeness_centrality(G0)
    #print(np.sum(list(closeness_centrality.values()), axis=0)/G0.number_of_nodes())
    
    
    # Shown the largest strongly connected component
    print('generating graph plot...')
    plt.figure()
    #ox.plot_graph(G0,figsize=(20, 20),node_color=list(closeness_centrality),bgcolor='w',edge_color='#111111')
    ox.plot_graph(G0,figsize=(20, 20))

    print('|V| = ',G0.number_of_nodes())
    print('|E| = ',G0.number_of_edges())
   

# Compute D_full, G_full, n_full, nodes
Gen_Lgraph = False
if Gen_Lgraph == True:
    D_full, S_full, n_full, nodes = Gen_road_network(G0)
    save('Data/D_full_PEK_5R.npy',D_full)
    save('Data/S_full_PEK_5R.npy',S_full)
    save('Data/nodes_PEK_5R.npy',nodes)
   
    

########################################################################################################



Gen_subgraph = False
if Gen_subgraph == True:
    
    city = 1
    if city == 0:
        D_full = load('Data/D_full_NY.npy', allow_pickle = True)
        S_full = load('Data/S_full_NY.npy', allow_pickle = True)
        nodes = load('Data/nodes_NY.npy', allow_pickle = True)
    elif city == 1:  
        D_full = load('Data/D_full_PEK_5R.npy', allow_pickle = True)
        S_full = load('Data/S_full_PEK_5R.npy', allow_pickle = True)
        nodes = load('Data/nodes_PEK_5R.npy', allow_pickle = True)     
    elif city == 2: 
        D_full = load('Data/D_full_PEK.npy', allow_pickle = True)
        S_full = load('Data/S_full_PEK.npy', allow_pickle = True)
        nodes = load('Data/nodes_PEK.npy', allow_pickle = True)
    elif city == 3:
        D_full = load('Data/D_full_LDN.npy', allow_pickle = True)
        S_full = load('Data/S_full_LDN.npy', allow_pickle = True)
        nodes = load('Data/nodes_LDN.npy', allow_pickle = True)
        
        

    n_full = np.shape(D_full)[0]
    
    A_full = A_meters(nodes)
    
                
    """
    Sample subgraphs of given sizes
    
    """
    Size_lst = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 80]
    N_Per_Size = 200 # 200
    #Size_lst = [3, 4, 5, 6, 7, 8, 9, 10]
    #N_Per_Size = 1 # 200    
    #Size_lst = [10]
    #N_Per_Size = 1
    
    
    graphs = Gen_subgraphs(D_full, S_full, A_full, n_full, nodes, Size_lst, N_Per_Size)
    N_graphs = len(graphs)
    
    
    # A_Full
    A_Full = graphs[0].A_full
    
    # S_Full
    S_Full = graphs[0].S_full
    w, v = LA.eig(S_Full)
    w = w.real
    w = np.sort(w)[::-1]
    lambda_1_S_Full = w[0]
    lambda_2_S_Full = w[1]

    
    # A_Sub, S_Sub and TSP_Y
    x_Data = np.zeros((N_graphs,1))
    y_Data = np.zeros(N_graphs)
    
    i=0
    for idx, graph in enumerate(graphs):
        
        if idx%10 == 0:
            print(idx)
        
        # Get Labels
        print('solving TSP')
        _, y_Data[idx] = TSP(graph.D_sub)


        # Get Features
        w, v = LA.eig(graph.S_sub)
        w = w.real
        w = np.sort(w)[::-1]
        lambda_1_S_Sub = w[0]
        lambda_2_S_Sub = w[1]
        sv_1_S_Sub = np.sqrt(lambda_1_S_Sub)
        sv_2_S_Sub = np.sqrt(lambda_2_S_Sub)
        v = v.T
        
        
        X = graph.Pos_nodes_sub
        X_mean = np.mean(X, axis=0)
        A = X - X_mean
        
        

       
                        
        A_est_PCA = sv_1_S_Sub*sv_2_S_Sub*4
        
        x_Data[idx,:] = np.sqrt( graph.N_nodes * np.array([A_est_PCA]) )
        
        plot_PCA = False
        if plot_PCA:
            plt.figure(figsize=(10,10))
            plt.scatter(A[:,0],A[:,1],s=10)
            plt.xlim([-8500,12500])
            plt.ylim([-6000,8000])
            
            plt.gca().set_aspect('equal', adjustable='box')
            x,y = 0,0
            dx,dy = sv_1_S_Sub*v[0,0],sv_1_S_Sub*v[0,1]
            plt.arrow(x,y,dx,dy,color = 'r',linewidth=5)
            
            x,y = 0,0
            dx,dy = sv_2_S_Sub*v[1,0],sv_2_S_Sub*v[1,1]
            plt.arrow(x,y,dx,dy,color = 'g',linewidth=5)

            plt.title('A_PCA = '+str(A_est_PCA))

        

            
    np.save('Output/Beijing_5R_2800_X_PCA_SV.npy',x_Data)    
    np.save('Output/Beijing_5R_2800_Y_PCA_SV.npy',y_Data)    
         

        
        
        
        