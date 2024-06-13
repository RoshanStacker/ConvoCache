from __future__ import annotations

import faiss
import numpy as np


class Cache:
    def __init__(
        self,
        index_type: str = "IP",
        embeddings: np.ndarray = None,
        responses: list = None,
        dimension_size: int = None,
        gpu: bool = False,
    ):
        self.index_type = index_type
        self.response_db = []  # responses are added in .add()
        self.gpu = gpu
        if embeddings is not None and responses is not None:
            self.index = self._create_index(index_type, embeddings.shape[1], gpu)
            self.add(embeddings, responses)
        elif dimension_size:
            self.index = self._create_index(index_type, dimension_size, gpu)
        else:
            raise ValueError(
                "Either embeddings and responses or dimension_size must be provided"
            )

    def _create_index(self, index_type: str, dimension_size, gpu=False) -> faiss.Index:
        """
        Create the index
        :param index_type: type of index to use,
                           - 'L2' for Euclidean distance,
                           - 'IP' for inner product (cosine similarity)
        :return: faiss index for searching
        """
        if self.index_type == "IP":  # Euclidean distance
            index = faiss.IndexFlatIP(dimension_size)
        elif self.index_type == "L2":  # Inner product (cosine similarity)
            index = faiss.IndexFlatL2(dimension_size)
        else:
            raise ValueError(f"Index type {self.index_type} not supported")
        if gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def add(self, embeddings: np.ndarray, responses):
        """
        Add embeddings and responses to the index

        :param embeddings: embeddings to add
        :param responses: responses to add. Can be a single string or a list of strings
        """
        # normalise embeddings
        faiss.normalize_L2(embeddings)  # Normalise the embeddings
        self.index.add(embeddings)
        if isinstance(responses, str):
            responses = [responses]
        self.response_db.extend(responses)

    def search(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        """
        Search the index for the k nearest neighbors
        :param query: query embeddings. Will be normalised
        :param k: number of nearest neighbors to find
        :return: responses, distances. In order of best to worst
        """
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        distances, indices = distances[0], indices[0]  # unpack the arrays
        # print(f"indices: {indices}\n len responses: {len(self.responses)}")
        responses = [self.response_db[i] for i in indices]
        return responses, distances


class ConvoCache:
    """A cache with an encoder, capable of completing dialogues"""

    def __init__(self, encoder, cache: Cache):
        self.encoder = encoder
        self.cache = cache

    def complete_dialogue(
        self, dialogue_history: list, k: int = 5
    ) -> tuple[list[str], list[float]]:
        """
        Find the k best responses for the given last_utterance
        :return: A tuple of lists of candidate responses and their scores. First is best.
                 For L2 Cache, low score is better. For IP Cache, high score is better.
        """
        query_embedding = self.encoder.encode(dialogue_history)
        return self.cache.search(query_embedding, k)
