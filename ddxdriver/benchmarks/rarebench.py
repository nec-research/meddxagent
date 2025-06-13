import json
from pathlib import Path
from typing import Dict, List, Mapping, Union
from datasets import Dataset, load_dataset

from ddxdriver.utils import Patient
from ddxdriver.logger import log

from ._fewshot_utils import make_and_save_embeddings, read_embeddings
from ._knn_search import KnnSearch
from .base import Bench

_DATASET_SUBSETS = [
    "RAMEDIS",
    "MME",
    "PUMCH_ADM",
]


class RareBench(Bench):
    """..."""

    # Redefine variable of base class
    DATASET_NAME = "rarebench"
    DIAGNOSIS_OPTIONS_FILE = "diagnosis_options.json"
    FEWSHOT_EMBEDDINGS_FILE = "embeddings"  # This is the folder, not the file
    ADD_1_FEWSHOT = True
    DDX_LENGTH = 10

    SPECIALIST_PREFACE = "You are a rare disease specialist."

    # Used to download the dataset
    DATASET_FILE = "chenxz/RareBench"

    def __init__(self, *args, dataset_subset: List[str] | None = None, **kwargs):
        # Define self.dataset_subset before super() init because needed in subclassed init functions
        self.dataset_subset = dataset_subset or _DATASET_SUBSETS
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Loads the dataset."""
        if not self.dataset_subset:
            raise ValueError(
                "Trying to load rarebench datasets with no subsets, returning early from load_dataset"
            )
        self._subsets_patient_entries: Dict[str, List[Patient]] = {}
        self._make_subsets_patient_entries()
        patient_keys = self._subsets_patient_entries[self.dataset_subset[0]][0].keys()
        combined_patients_list = [
            patient
            for patient_list in self._subsets_patient_entries.values()
            for patient in patient_list
        ]
        data_dict = {key: [dic[key] for dic in combined_patients_list] for key in patient_keys}
        dataset = Dataset.from_dict(data_dict)
        return dataset

    def load_disease_mapping(self) -> Dict[str, str]:
        """..."""
        disease_mapping_path = self._dataset_path / self.DISEASE_MAPPING_FILE
        with disease_mapping_path.open() as fp:
            disease_mapping_json = json.load(fp)

        concatenated_mapping = {}
        for subset in self.dataset_subset:
            if subset not in disease_mapping_json:
                raise ValueError(f"Dataset subset {subset} not in mapping")

            for disease_codes, disease in disease_mapping_json[subset].items():
                concatenated_mapping[disease_codes] = disease

        return concatenated_mapping

    def load_diagnosis_options(self) -> List[str]:
        """Loads the available diagnosis options."""
        diagnosis_options_path = self._dataset_path / self.DIAGNOSIS_OPTIONS_FILE
        diagnosis_options_json = json.load(diagnosis_options_path.open())

        diagnosis_options = set()
        for subset in self.dataset_subset:
            diagnosis_options.update(diagnosis_options_json[subset])

        return list(diagnosis_options)

    def get_patient_profile(self, patient_entry: Mapping) -> str:
        """Returns the (full) patient profile."""
        return "\n".join("- " + symptom for symptom in patient_entry["phenotypes"].split(", "))

    def get_patient_initial_info(self, patient_entry: Mapping) -> str:
        """Returns ~20% of the patient's symptoms (minimum 1)."""
        symptoms = patient_entry["phenotypes"].split(", ")
        num_symptoms = max(1, round(len(symptoms) * 0.2))  # at least 1 symptom
        initial_symptoms = symptoms[:num_symptoms]
        return "\n".join("- " + symptom for symptom in initial_symptoms)

    def get_patient_gt_pathology(self, patient_entry: Mapping) -> str:
        """Returns the ground truth pathology for the given patient."""
        pathology = patient_entry["pathology"]
        if pathology not in self.disease_mapping:
            raise ValueError(
                f"Rarebench: Could not find ground truth disease for patient pathology: {pathology}"
            )
        return self.disease_mapping[pathology]

    def get_patient_gt_ddx(self, patient_entry: Mapping) -> List[str]:
        """Returns the ground truth ddx for the given patient."""
        return []

    def make_fewshot_embeddings(self, knn_search: KnnSearch, filename: Union[str, Path]):
        """..."""
        log.info("Creating fewshot embeddings...\n")
        filename = Path(filename)
        # Ensure the embeddings directory exists
        filename.mkdir(parents=True, exist_ok=True)
        for subset in self.dataset_subset:
            patients = [
                self.create_patient(patient_entry)
                for patient_entry in self._subsets_patient_entries[subset]
            ]
            subset_file = filename / f"{subset}_embeddings.json"
            make_and_save_embeddings(patients, knn_search, subset_file)

    def parse_fewshot_embeddings_file(
        self, knn_search: KnnSearch, filename: Union[str, Path]
    ) -> KnnSearch:
        """..."""
        filename = Path(filename)
        # Ensure the embeddings directory exists
        filename.mkdir(parents=True, exist_ok=True)
        for subset in self.dataset_subset:
            subset_file = filename / f"{subset}_embeddings.json"

            if not subset_file.exists():
                patients = [
                    self.create_patient(patient_entry)
                    for patient_entry in self._subsets_patient_entries[subset]
                ]
                make_and_save_embeddings(patients, knn_search, subset_file)

            knn_search = read_embeddings(knn_search, subset_file)
        return knn_search

    # Some utilities

    def _make_subsets_patient_entries(self):
        """..."""
        for subset in self.dataset_subset:
            dataset_subset = load_dataset(self.DATASET_FILE, subset, split="test", revision="4bb064a52a253cfaa5ee228926e86af3ec0e5731")
            patient_entries = self._load_ehr_phenotype_data(dataset_subset)
            for i, patient_entry in enumerate(patient_entries):
                patient_entry["id"] = f"{subset}_{i}"
            self._subsets_patient_entries[subset] = patient_entries

    def _load_ehr_phenotype_data(self, dataset: Dataset) -> List[Dict[str, str]]:
        """..."""
        # Loads phenotype and disease mappings
        phenotype_map_file = self._dataset_path / "rarebench_phenotype_mapping.json"
        disease_map_file = self._dataset_path / "rarebench_disease_mapping.json"
        phenotype_mapping = json.load(phenotype_map_file.open(encoding="utf-8-sig"))
        disease_mapping = json.load(disease_map_file.open(encoding="utf-8-sig"))

        patient_entries = []
        for patient_entry in dataset:
            phenotype_list = [
                phenotype_mapping[phenotype]
                for phenotype in patient_entry["Phenotype"]
                if phenotype in phenotype_mapping
            ]
            disease_list = [
                disease_mapping[disease]
                for disease in patient_entry["RareDisease"]
                if disease in disease_mapping
            ]
            phenotype = ", ".join(phenotype_list)
            disease = ", ".join(disease_list)
            if not phenotype or not disease:
                continue
            patient_entries.append({"phenotypes": phenotype, "pathology": disease})

        return patient_entries
