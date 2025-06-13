from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Union
from datasets import Dataset
import random

from ddxdriver.utils import parse_differential_diagnosis, Patient, Constants
from ddxdriver.logger import log

from ._fewshot_utils import make_and_save_embeddings, read_embeddings
from ._knn_search import KnnSearch
from .metrics import get_metrics

# Some types and constants
CONFIG = Dict[str, Any]

_data_path = Path(__file__).parent / "data"

class Bench(ABC):
    """
    Base class for benchmark datasets.

    Methods that MUST be subclassed:
    - load_dataset
    - get_patient_profile
    - get_patient_initial_info
    - get_patient_gt_pathology
    - get_patient_gt_ddx

    Methods that MIGHT be subclassed:
    - load_disease_mapping
    - load_patients
    - load_diagnosis_options
    - get_patient_id
    - get_fewshot_dataset
    - make_fewshot_embeddings
    - parse_fewshot_embeddings_file
    - create_patient
    """

    # These MUST be redefined in the concrete classes
    DATASET_NAME: str = ""
    SPECIALIST_PREFACE: str = "You are a medical doctor."

    # These MIGHT be redefined in the concrete classes
    DIAGNOSIS_OPTIONS_FILE = "diagnosis_options.txt"
    DISEASE_MAPPING_FILE = "disease_mapping.json"
    FEWSHOT_EMBEDDINGS_FILE = "fewshot_embeddings.json"
    ADD_1_FEWSHOT = False

    DDX_LENGTH = None

    def __init__(
        self, enforce_diagnosis_options: bool = True, knn_search_cfg: Optional[CONFIG] = None
    ):
        self._dataset_path = _data_path / self.DATASET_NAME
        self.disease_mapping = self.load_disease_mapping()
        self.dataset = self.load_dataset()
        self.patients = self.load_patients()

        # Diagnosis options might not be loaded. Load here since needs some paths defined above
        self.diagnosis_options = []
        if enforce_diagnosis_options:
            self.diagnosis_options = self.load_diagnosis_options()
        # Index and config for dynamic fewshot (lazy load)
        self.fewshot_index = None
        self.knn_search_cfg = knn_search_cfg

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Loads the dataset."""
        raise NotImplementedError

    def load_disease_mapping(self) -> Dict[str, str]:
        """
        If overloaded, can return a dictionary which maps from the ground truth pathology in the dataset to the diagnosis option actually used
        """

    def load_patients(self) -> List[Patient]:
        """
        From the dataset in load_dataset, plus precomputed diagnosis_options.t
        """
        if not self.dataset:
            raise ValueError("Trying to load patients without first loading dataset")
        try:
            patients = [
                self.create_patient(patient_entry=patient_entry) for patient_entry in self.dataset
            ]
            return patients
        except Exception as e:
            raise Exception(f"Error loading patients:\n{e}")

    def load_diagnosis_options(self) -> List[str]:
        """
        Loads the available diagnosis options. Assumes the diagnosis options are
        saved in a txt file, one option per line.

        In case the format is different, override this method.
        """
        diagnosis_options_path = self._dataset_path / self.DIAGNOSIS_OPTIONS_FILE
        diagnosis_options = diagnosis_options_path.read_text().split("\n")
        return diagnosis_options

    @abstractmethod
    def get_patient_profile(self, patient_entry: Mapping) -> str:
        """
        Returns the (full) patient profile. This should contain all relevant input
        data for a patient (antecedents, symptoms, age, sex, etc...).
        """

    @abstractmethod
    def get_patient_initial_info(self, patient_entry: Mapping) -> str:
        """Returns the initial patient information."""

    @abstractmethod
    def get_patient_gt_pathology(self, patient_entry: Mapping) -> str:
        """Returns the ground truth pathology for the given patient."""

    @abstractmethod
    def get_patient_gt_ddx(self, patient_entry: Mapping) -> List[str]:
        """Returns the ground truth ddx for the given patient."""

    def get_patient_id(self, patient_entry) -> Any:
        """
        Returns an id for the patient to check for equality
        This id should be not None and have __eq__ and __ne__ defined for == and != comparison
        """
        return patient_entry.get("id", None)

    def get_fewshot_dataset(self) -> List[Patient]:
        """
        Returns a list of patients which can be sampled to find fewshot examples.

        By default, the entire patients list is used.
        """
        return self.patients

    def get_fewshot(
        self,
        patient: Patient,
        fewshot_cfg: CONFIG,
    ) -> List[Patient]:
        """
        Returns a List of Patient fewshot examples

        The shots should be dictionaries with the following keys:
        - patient_profile
        - gt_pathology (ground truth pathology)
        - gt_ddx (ground truth differential diagnosis, optional)
        """

        # Check number of shots
        num_shots = fewshot_cfg.get("num_shots", 0)
        if num_shots < 0:
            log.warning("Requesting negative number of shots... Will return zero shots...")
            return []
        if num_shots == 0:
            return []

        match fewshot_cfg["type"].lower():
            case "static":
                return self.get_static_fewshot(patient, num_shots, fewshot_cfg)
            case "dynamic":
                return self.get_dynamic_fewshot(patient, num_shots)
            case "none":
                return []
            case _:
                log.warning(
                    f"Unknown fewshot type {fewshot_cfg['type']}... Will return zero shots..."
                )
                return []

    def get_static_fewshot(
        self,
        patient: Patient,
        num_shots: int,
        fewshot_cfg: CONFIG,
    ) -> List[Patient]:
        """Uniformly samples the few-shot examples from the dataset."""
        # Gets pool dataset
        fewshot_patient_pool = self.get_fewshot_dataset()
        seed = Constants.SEED.value
        if seed:
            random.seed(seed)
            random.shuffle(fewshot_patient_pool)

        shots = []
        for fewshot_patient in fewshot_patient_pool:
            if len(shots) == num_shots:
                break
            if fewshot_patient != patient:
                shots.append(fewshot_patient)
        return shots

    def get_dynamic_fewshot(
        self,
        patient: Patient,
        num_shots: int,
    ) -> List[Patient]:
        """Samples the most similar examples from the fewshot dataset."""
        if self.fewshot_index is None:
            self.fewshot_index = self.load_fewshot_index()

        top_k = num_shots + int(self.ADD_1_FEWSHOT)
        if not patient or not isinstance(patient.patient_profile, str):
            raise ValueError(
                "Trying to embed patient profile for dynamic fewshot but patient is None or its profile isn't a string"
            )
        fewshot_packed_patients = self.fewshot_index.retrieve(
            query=patient.patient_profile, top_k=top_k
        )
        shots = []
        for fewshot_packed_patient in fewshot_packed_patients:
            if len(shots) == num_shots:
                break
            fewshot_patient = Patient(**fewshot_packed_patient)
            if fewshot_patient != patient:
                shots.append(fewshot_patient)
        return shots

    def load_fewshot_index(self) -> KnnSearch | None:
        """Loads the fewshot index from disk."""
        # Check if file exists or if embeddings should be computed again
        fewshot_path = self._dataset_path / self.FEWSHOT_EMBEDDINGS_FILE
        if not fewshot_path.exists() or (
            self.knn_search_cfg is not None
            and self.knn_search_cfg.get("precompute_new_embeddings", False)
        ):
            knn_search = KnnSearch(knn_config=self.knn_search_cfg)
            self.make_fewshot_embeddings(knn_search, fewshot_path)
        # Load index from file
        knn_search = KnnSearch(knn_config=self.knn_search_cfg)
        return self.parse_fewshot_embeddings_file(knn_search, fewshot_path)

    def make_fewshot_embeddings(self, knn_search: KnnSearch, filename: Union[str, Path]):
        """
        Computes the fewshot embeddings and save them on file. By default, embeds all the patients
        in the fewshot dataset, saving the embedding alongside the original patient entry.

        Args:
            knn_search: KnnSearch object
            filename (path-like): File where embeddings should be saved
        """
        log.info("Creating fewshot embeddings...\n")
        make_and_save_embeddings(self.patients, knn_search, filename)

    def parse_fewshot_embeddings_file(
        self, knn_search: KnnSearch, filename: Union[str, Path]
    ) -> KnnSearch:
        """Load the fewshot embeddings from a file."""
        return read_embeddings(knn_search, filename)

    def create_patient(self, patient_entry: Mapping) -> Patient:
        return Patient(
            patient_id=self.get_patient_id(patient_entry),
            patient_initial_info=self.get_patient_initial_info(patient_entry),
            gt_pathology=self.get_patient_gt_pathology(patient_entry),
            patient_profile=self.get_patient_profile(patient_entry),
            gt_ddx=self.get_patient_gt_ddx(patient_entry),
        )
