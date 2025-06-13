import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
import numpy as np

from ddxdriver.utils import Patient
from ._knn_search import KnnSearch


def make_and_save_embeddings(
    patients: List[Patient],
    knn_search: KnnSearch,
    filename: Union[str, Path],
):
    """
    Computes and save the embeddings of the given patient entries. The embeddings, alongside the
    original patient info (packed into a dictionary), are saved into a json file.

    Args:
        patient_entries: List of Patient's containing the patient entries to embed
        knn_search: KnnSearch object. Used to encode the data
        filename: File where to save the embeddings
    """
    embeddings_patients: List[Tuple[List[float], Patient]] = []
    for patient in patients:
        embedding = knn_search.encode_data(patient.patient_profile)
        embeddings_patients.append((embedding.tolist(), patient.pack_attributes()))

    filename = Path(filename)
    with filename.open("w") as fp:
        json.dump(embeddings_patients, fp, indent=4)


def read_embeddings(knn_search: KnnSearch, filename: Union[str, Path]) -> KnnSearch:
    """
    Reads the embeddings from file and inserts them into the given KnnSearch object.
    The files are assumed to be generated with `meth:make_and_save_embeddings`.

    Args:
        knn_search: KnnSearch object. This will be updated
        filename: File where the embeddings are saved

    Return:
        Modified KnnSearch object
    """
    filename = Path(filename)
    with filename.open() as fp:
        json_content = json.load(fp)

    patient_profile_embeddings = [np.array(embedding) for embedding, _ in json_content]
    patients = [patient for _, patient in json_content]

    knn_search.insert_precomputed(
        embeddings=patient_profile_embeddings,
        metadata=patients,
    )
    return knn_search
