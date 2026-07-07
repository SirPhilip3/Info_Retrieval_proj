from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin
import networkx as nx
import numpy as np


def create_sparse_tf_idf_matrix(index, dtype=np.float32):

    """
    Input: 
        index: inverted index
    Ouput: 
        Matrix of tf_idf for each document
        List of doc_ids
        List of term
    """


    terms = list(index.get_terms())
    term_to_col = {t:i for i,t in enumerate(terms)}
    doc_ids = index.get_doc_ids()
    doc_to_row = {did:i for i,did in enumerate(doc_ids)}

    rows, cols, data = [], [], []
    for term, posting in index.index.items():
        idf = index.get_inverse_document_frequency(term)
        col = term_to_col[term]
        for doc_id, tf in posting:
            rows.append(doc_to_row[doc_id])
            cols.append(col)
            data.append(np.log10(1+tf) * idf)

    data = np.array(data, dtype=dtype)
    mat = csr_matrix((data, (rows, cols)), shape=(len(doc_ids), len(terms)), dtype=dtype)
    return mat, doc_ids, terms


def k_medoids(df, k, verbose = False): # TODO vary for k vs execution time!

    """ 
    input:
        df: dataset
        k: number of clusters
    output:
        medoids
        list of medoid -> docId that belong to the same cluster
    """
    norm_df = normalize(df, norm="l2")

    model = KMeans(n_clusters=k, random_state=0)
    model.fit(norm_df)

    # cluster of each document
    labels = model.labels_

    medoids_id = []

    for i in range(model.n_clusters):
        cluster_docs = np.flatnonzero(labels == i)
        cluster_vectors = norm_df[cluster_docs]
        
        centroid = model.cluster_centers_[i].reshape(1, -1)

        closest = pairwise_distances_argmin(cluster_vectors, centroid, metric='euclidean')[0]

        medoids_id.append(cluster_docs[closest])

    # medoid_vectors = norm_df[medoids].toarray()

    medoid_vectors = norm_df[medoids_id]
    nonzero_cols = medoid_vectors.getnnz(axis=0) > 0
    medoid_vectors = medoid_vectors[:, nonzero_cols].toarray()

    cluster_to_docs = {}

    for cluster_id, medoid_doc in enumerate(medoids_id):
        cluster_to_docs[medoid_doc] = np.flatnonzero(labels == cluster_id).tolist()

    if verbose:
        print("Medoid DocIDs:", medoids_id)

    return medoid_vectors, medoids_id, cluster_to_docs




def stream_cluster(SortedCollection, Radius, k): 

    """
    input: 
        SortedCollection: the document-term mapping ordered by the number of the term for each doc
        Radius: minimum distance between documents to be considered close
        k: maximum number of clusters 
    output: 
        C = a list that contain the doc_id of the Medoids
        Cluster = a list of medoid_id -> list of doc_id belong to the same cluster
    
    """


    C = []                       # Medoids list
    medoid_sets = {}             # medoid_id -> terms set of that medoid
    Clusters = defaultdict(list) # medoid_id -> list of doc_id belong to the same cluster

    for doc_id, doc_terms in SortedCollection.items():
        set_d = set(doc_terms)
        
        # If there are no medoids yet, create the first cluster with this document as the medoid
        if not C:
            C.append(doc_id)
            medoid_sets[doc_id] = set_d
            Clusters[doc_id].append(doc_id)
            continue

        # Calculate distances from all current medoids
        distances = []
        for c in C:
            centroid_set = medoid_sets[c]
            intersection = len(centroid_set.intersection(set_d))
            union = len(centroid_set.union(set_d))
            jaccard_dist = 1 - (intersection / union) if union > 0 else 1
            distances.append(jaccard_dist)
        
        # Find the closest medoid
        min_idx = np.argmin(distances)
        dist_c = distances[min_idx]
        closest_medoid = C[min_idx]

        # If it is close enough OR we have reached the maximum number of clusters k
        if dist_c < Radius or len(C) >= k:
            Clusters[closest_medoid].append(doc_id)
        else:
            # Create a new cluster with this document as the medoid
            C.append(doc_id)
            medoid_sets[doc_id] = set_d
            Clusters[doc_id].append(doc_id)
            
    return C,Clusters



def calculate_distance_matrix(medoid_vectors, method="euclidean"):

    """ 
    input: 
        medoid_vectors: list of medoid_vectors    
        method: "euclidea", "cosine", "jaccard"
    output: 
        pairwise_distance (for the jaccardian returns directly the distance matrix)
    """


    if method == "euclidean":
        return pairwise_distances(medoid_vectors, metric="euclidean")
    elif method == "cosine":
        return pairwise_distances(medoid_vectors, metric="cosine")
    elif method == "jaccard":
        return pairwise_distances((medoid_vectors > 0).astype(np.uint8), metric="jaccard")
    elif method == "jaccard_optimized":

        #  calculate only the upper triangular matrix
        n = len(medoid_vectors)
        dist = np.zeros((n, n))
        
        # Trasformiamo in set una volta sola prima dei cicli
        sets = [set(v) for v in medoid_vectors]

        for i in range(n):
            for j in range(i + 1, n):  # Parte da i+1 per sfruttare la simmetria
                intersection = len(sets[i].intersection(sets[j]))
                union = len(sets[i].union(sets[j]))
                
                jaccard_dist = 1 - (intersection / union) if union > 0 else 1
                
                dist[i][j] = jaccard_dist
                dist[j][i] = jaccard_dist  # Specchia il risultato
                
        return dist
    
    else:
        raise ValueError(f"Unknown method: {method}")
    


def tsp_path(dist_matrix, verbose=False):

    """
    Input: 
        Distance matrix of the medoid graph
    Output:
        Tsp path
    """

    tsp = nx.approximation.traveling_salesman_problem

    tsp_graph = nx.from_numpy_array(dist_matrix)
    tsp_path = tsp(tsp_graph, cycle=False)

    if verbose:
        print("TSP path (medoid indices):", tsp_path)

    return tsp_path

# def stream_cluster(SortedCollection, Radius, k): 
#     """
#     Versione modificata per mantenere gli indici posizionali (0, 1, 2...) 
#     compatibili con la matrice delle distanze e il codice TSP esistente.
#     """
#     C_ids = []               # Lista dei REALI doc_id dei medoidi (es: ['doc_42', 'doc_99'])
#     medoid_sets = []         # Lista dei set di termini (l'indice corrisponde alla posizione nella matrice)
    
#     # I cluster ora usano come chiave l'INDICE POSIZIONALE (0, 1, 2...) e non il doc_id reale
#     Clusters = defaultdict(list) 

#     for doc_id, doc_terms in SortedCollection.items():
#         set_d = set(doc_terms)
        
#         # Se non ci sono ancora medoidi, crea il primo (indice 0)
#         if not C_ids:
#             C_ids.append(doc_id)
#             medoid_sets.append(set_d)
#             Clusters[0].append(doc_id)
#             continue

#         # Calcola le distanze usando l'indice posizionale 'idx'
#         distances = []
#         for idx in range(len(C_ids)):
#             centroid_set = medoid_sets[idx]
#             intersection = len(centroid_set.intersection(set_d))
#             union = len(centroid_set.union(set_d))
#             jaccard_dist = 1 - (intersection / union) if union > 0 else 1
#             distances.append(jaccard_dist)
        
#         # Trova il medoide più vicino (min_idx sarà 0, 1, 2...)
#         min_idx = np.argmin(distances)
#         dist_c = distances[min_idx]

#         # Se è vicino OR abbiamo raggiunto il tetto massimo k
#         if dist_c < Radius or len(C_ids) >= k:
#             # Assegna al cluster usando l'indice posizionale
#             Clusters[min_idx].append(doc_id)
#         else:
#             # Crea un nuovo medoide (il suo indice sarà il vecchio len(C_ids))
#             new_idx = len(C_ids)
#             C_ids.append(doc_id)
#             medoid_sets.append(set_d)
#             Clusters[new_idx].append(doc_id)
            
#     # Restituiamo medoid_sets (lista di set per la distanza) e C_ids come terzo parametro
#     return medoid_sets, Clusters, C_ids
    
# def reorder_documents_by_tsp_stream_cluster(document_clusters, medoids, tsp_path):
#     """
#     Reorder documents by cluster order induced by TSP over medoids.
#     Inside each cluster, keep documents in their original order (row order).

#     Args:
#         document_clusters: array-like of shape (n_docs,), cluster label per doc.
#         medoids: list of medoid doc indices (not required for ordering here, kept for API compatibility).
#         tsp_path: list of cluster ids in the order returned by TSP.

#     Returns:
#         ordered_doc_ids: list of document row indices in final order.
#         cluster_to_docs: dict {cluster_id: [doc_ids in original order]}.
#     """
#     document_clusters = np.asarray(document_clusters)

#     # Build cluster -> docs preserving natural/original order
#     cluster_to_docs = {
#         c: np.flatnonzero(document_clusters == c).tolist()
#         for c in range(len(medoids))
#     }

#     # Concatenate clusters following TSP order
#     ordered_doc_ids = []
#     for c in tsp_path:
#         ordered_doc_ids.extend(cluster_to_docs.get(c, []))

#     return ordered_doc_ids, cluster_to_docs

# better 
def reorder_documents_by_tsp(tsp_path, cluster, C):
    tot = []
    for i in tsp_path:
        # retrieve the real DocID of the medoid using the TSP index
        medoid_id = C[i] 
        
        docs = cluster.get(medoid_id, [])
        
        for d in docs:  
            tot.append(d)
        
    return tot

