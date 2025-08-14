import random
from functools import lru_cache
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def diverse_sampling(data_list, k, num_clusters, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Selects a diverse subset of items from a list of documents.

    This function first embeds all documents into vector representations using a
    sentence transformer model. It then clusters these embeddings into a specified
    number of clusters using k-means. Finally, it samples a number of documents
    from each cluster using a method that promotes diversity based on cosine
    similarity.

    Args:
        data_list (list[str]): A list of documents to sample from.
        k (int): The total number of samples to choose.
        num_clusters (int): The number of clusters to group the documents into.
        model_name (str): Model to use for embedding the documents. Default is
                          "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        list[int]: A list of indices of the selected diverse samples from the
                   original data_list.
    """
    # 1. Embed all documents
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        data_list,
        convert_to_tensor=False,
        show_progress_bar=True,
    )

    # 2. Cluster the embeddings into num_clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create a dictionary to store the indices of docs in each cluster
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)

    # 3. Pick k/num_clusters docs per cluster
    samples_per_cluster = k // num_clusters
    selected_indices = []

    @lru_cache(maxsize=None)
    def get_cosine_similarity(index1, index2):
        """
        Calculates the cosine similarity between two embeddings, with LRU caching.
        """
        return cosine_similarity([embeddings[index1]], [embeddings[index2]])[0][0]

    for cluster_id in range(num_clusters):
        cluster_indices = clusters[cluster_id]
        if not cluster_indices:
            continue

        # Initial shuffling of the list
        random.shuffle(cluster_indices)

        # Start with the first item
        cluster_selected = [cluster_indices.pop(0)]

        # Set initial similarity threshold
        threshold = 0.05

        while len(cluster_selected) < samples_per_cluster and cluster_indices:
            found_a_sample = False
            items_to_remove_from_cluster_indices = []

            for i, sample_index in enumerate(cluster_indices):
                is_dissimilar_to_all = True
                for selected_index in cluster_selected:
                    if get_cosine_similarity(sample_index, selected_index) >= threshold:
                        is_dissimilar_to_all = False
                        break
                
                if is_dissimilar_to_all:
                    cluster_selected.append(sample_index)
                    items_to_remove_from_cluster_indices.append(i)
                    found_a_sample = True
                    if len(cluster_selected) >= samples_per_cluster:
                        break
            
            # Remove selected items from the pool for this cluster
            for i in sorted(items_to_remove_from_cluster_indices, reverse=True):
                del cluster_indices[i]

            # If no new sample was added in this pass, increase the threshold
            if not found_a_sample:
                threshold += 0.05
        
        selected_indices.extend(cluster_selected)

    return selected_indices

if __name__ == '__main__':
    # Example Usage
    documents = [
        "The cat sat on the mat.",
        "A feline was resting on the rug.",
        "Dogs are loyal companions.",
        "Canines make great pets.",
        "The sun is shining brightly today.",
        "It's a beautiful sunny day.",
        "Artificial intelligence is a fascinating field.",
        "Machine learning is a subset of AI.",
        "The stock market is volatile.",
        "Investing requires careful research.",
        "Baking a cake is a fun activity.",
        "Let's bake a delicious chocolate cake."
    ]

    # Initialize a sentence transformer model
    # For this example, we use a pre-trained model. 
    # You can choose any model from sentence-transformers documentation. [5, 7]
    try:
        transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Failed to load the model. Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`). Error: {e}")
        transformer_model = None

    if transformer_model:
        # Number of samples to select
        k_samples = 6
        # Number of clusters to create
        n_clusters = 3

        print(f"Original number of documents: {len(documents)}")
        print(f"Number of samples to select: {k_samples}")
        print(f"Number of clusters: {n_clusters}\n")

        # Get the diverse samples
        diverse_sample_indices = diverse_sampling(documents, k_samples, n_clusters, transformer_model)

        print("Selected diverse sample indices:")
        print(diverse_sample_indices)

        print("\nSelected diverse documents:")
        for index in diverse_sample_indices:
            print(f"- {documents[index]}")