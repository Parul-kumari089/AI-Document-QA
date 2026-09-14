import faiss
import numpy as np


def create_vector_store(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_vector_store(
    index,
    query_embedding,
    k=3
):

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances, indices


# SAVE FAISS INDEX
def save_vector_store(
    index,
    file_path
):

    faiss.write_index(
        index,
        file_path
    )


# LOAD FAISS INDEX
def load_vector_store(
    file_path
):

    index = faiss.read_index(
        file_path
    )

    return index