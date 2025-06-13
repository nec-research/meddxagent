import random
from enum import Enum
from typing import Dict, List

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class RetrieveOrder(Enum):
    SIMILAR_AT_TOP = "similar_at_top"
    SIMILAR_AT_BOTTOM = "similar_at_bottom"
    RANDOM = "random"


class Pooling(Enum):
    CLS = "cls"
    AVERAGE = "average"
    MAX = "max"


_DEFAULT_KNN_CFG = {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "pooling": "cls",
    "order": "similar_at_top",
    "seed": None,
}


class KnnSearch:
    def __init__(self, knn_config: Dict | None) -> None:
        self.cfg = _DEFAULT_KNN_CFG
        self.cfg.update(knn_config or {})

        embedding_model = self.cfg["embedding_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model).eval()
        self.index = None
        self.pooling = self.cfg["pooling"]
        pooling_types = {member.value for member in Pooling}
        assert self.pooling in pooling_types, "Chose unsupported pooling type"
        self.max_length = 512
        self.id2evidence = dict()
        self.embed_dim = len(self.encode_data("Test embedding size"))
        self.insert_acc = 0
        self.seed = self.cfg["seed"]

        self.retrieve_order = self.cfg["order"]
        orders = {member.value for member in RetrieveOrder}
        assert self.retrieve_order in orders, "Chose unsupported ordering"
        self.create_faiss_index()

    def create_faiss_index(self):
        self.index = faiss.IndexFlatL2(self.embed_dim)

    def encode_data(self, sentence: str) -> np.ndarray:
        encoded_input = self.tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)

        if self.pooling == Pooling.CLS.value:
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        elif self.pooling == Pooling.AVERAGE.value:
            sentence_embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
        elif self.pooling == Pooling.MAX.value:
            sentence_embeddings = model_output.last_hidden_state.max(dim=1).values.numpy()
        else:
            raise ValueError("Invalid pooling strategy")

        feature = sentence_embeddings[0]
        norm = np.linalg.norm(feature)
        return feature / norm

    def insert(self, key: str, value: str) -> None:
        embedding = self.encode_data(key).astype("float32")
        self.index.add(np.expand_dims(embedding, axis=0))
        self.id2evidence[str(self.insert_acc)] = value
        self.insert_acc += 1

    def insert_precomputed(self, embeddings: List[np.ndarray], metadata: List) -> None:
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.id2evidence.update(
            {str(i + len(self.id2evidence)): metadata[i] for i in range(len(metadata))}
        )
        self.insert_acc += len(metadata)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        embedding = self.encode_data(query).astype("float32")
        top_k = min(top_k, self.insert_acc)
        distances, indices = self.index.search(np.expand_dims(embedding, axis=0), top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        results = [
            {"link": str(idx), "_score": {"faiss": dist}} for dist, idx in zip(distances, indices)
        ]
        if self.retrieve_order == RetrieveOrder.SIMILAR_AT_BOTTOM.value:
            results = list(reversed(results))
        elif self.retrieve_order == RetrieveOrder.RANDOM.value:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(results)

        text_list = [self.id2evidence[result["link"]] for result in results]
        return text_list


if __name__ == "__main__":
    knn_config = {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "seed": 42,
        "order": "similar_at_top",  # ["similar_at_top", "similar_at_bottom", "random"]
        "pooling": None,
    }

    patient_profiles = [
        "Patient with hypertension and diabetes, aged 60",
        "Patient with asthma, aged 25",
        "Patient with chronic back pain, aged 45",
        "Patient with history of stroke, aged 70",
        "Patient with hypertension and high cholesterol, aged 55",
        "Patient with diabetes and obesity, aged 50",
        "Patient with mild cognitive impairment, aged 80",
        "Patient with seasonal allergies, aged 30",
    ]

    query = "Patient with diabetes and hypertension, aged 65"

    pooling_methods = ["cls", "average", "max"]

    for pooling in pooling_methods:
        knn_config["pooling"] = pooling
        print(f"\nPooling method: {pooling}")
        knn_search = KnnSearch(knn_config)

        for profile in patient_profiles:
            knn_search.insert(key=profile, value=profile)

        results = knn_search.retrieve(query, top_k=5)
        print(f"Results for {pooling} pooling:")
        for result in results:
            print(result)
