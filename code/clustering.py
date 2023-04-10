from create_data import prepare_for_clustering
import networkx as nx
from numpy import linalg as LA
import math
from community import community_louvain
import numpy as np
import pandas as pd


def compute_C_minus_C0(lambdas,v,lambda_plus):
    N = len(lambdas)
    C_clean = np.zeros((N, N))
    v_m = np.matrix(v)
    for i in range(N):
        if lambdas[i] > lambda_plus:
            C_clean = C_clean + lambdas[i] * np.dot(v_m[:, i], v_m[:, i].T)  
    return C_clean    
    

def LouvainCorrelationClustering(R):
    N=R.shape[1]
    T=R.shape[0]

    q = N * 1. / T
    lambda_plus = (1. + np.sqrt(q)) ** 2

    C = R.corr()
    lambdas, v = LA.eigh(C)

    order = np.argsort(lambdas)
    lambdas,v = lambdas[order], v[:,order]
    
    C_s = compute_C_minus_C0(lambdas, v, lambda_plus)
    
    mygraph = nx.from_numpy_array(np.abs(C_s))
    partition = community_louvain.best_partition(mygraph)

    values = [partition.get(node) for node in mygraph.nodes()]
    print("Total number of clusters: {}".format(len(np.unique(np.array(values)))))

    DF = pd.DataFrame.from_dict(partition, orient="index")

    return DF, mygraph, values


def add_clusters(df, show=False):
    """
    Compute clusters.
    """
    # Get data format suitable for clustering
    df_cluster = prepare_for_clustering(df.copy(), show)

    # Choose part of data to cluster
    T_in = 3 * df_cluster.shape[1]
    t0 = df_cluster.shape[0] - T_in
    t1 = t0 + T_in

    # Select data
    df_cluster_cut = df_cluster.iloc[t0:t1]
    sel = df_cluster_cut.isnull().sum(axis=0) > 0
    df_cluster_cut = df_cluster_cut.drop(columns=df_cluster_cut.columns[sel])
    print("Data shape for clustering is {}".format(df_cluster_cut.shape))

    # Compute clusters
    df_louvain, graph_louvain, graph_values = LouvainCorrelationClustering(df_cluster_cut)
    print(df_louvain.value_counts())

    # Add cluster id as feature, for the clustered tickers
    df_with_clusters = df[df['Name'].apply(lambda x: x in list(df_cluster_cut.columns))]
    stocks_cluster = pd.DataFrame([list(df_cluster_cut.columns), graph_values]).T.rename(columns={0: 'Name', 1: 'Cluster'})
    df_with_clusters = df_with_clusters.merge(stocks_cluster, on='Name')
    df_with_clusters['Cluster'] = df_with_clusters['Cluster'].astype(int)

    return df_with_clusters

