import json
from typing import List, Mapping, Dict
from datasets import Dataset

from ddxdriver.utils import strip_all_lines, Patient

from .base import Bench


class ICraftMD(Bench):
    """..."""

    # Redefine variable of base class
    DATASET_NAME = "icraftmd"
    SPECIALIST_PREFACE = "You are a skin disease specialist."
    ADD_1_FEWSHOT = True
    DDX_LENGTH = 10

    # Used to load the dataset
    DATASET_FILE = "all_craft_md.jsonl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Loads the dataset."""
        dataset_file_path = self._dataset_path / self.DATASET_FILE
        with dataset_file_path.open() as fp:
            json_list = [json.loads(line) for line in fp]
        return Dataset.from_list(json_list)

    def load_disease_mapping(self) -> Dict[str, str]:
        """..."""
        disease_mapping_path = self._dataset_path / self.DISEASE_MAPPING_FILE
        with disease_mapping_path.open() as fp:
            disease_mapping_json = json.load(fp)

        # Should already be in dictionary mapping format (no subsets), just return
        return disease_mapping_json

    def get_patient_profile(self, patient_entry: Mapping) -> str:
        """Returns the (full) patient profile."""
        return "\n".join("- " + phenotype for phenotype in patient_entry["context"])

    def get_patient_initial_info(self, patient_entry: Mapping) -> str:
        """Returns the initial patient information."""
        return strip_all_lines(
            f"""\
            Age: {patient_entry["patient"]["age"]}
            Sex: {patient_entry["patient"]["gender"]}
            Chief Complaint: {patient_entry["context"][0]}"""
        )

    def get_patient_gt_pathology(self, patient_entry: Mapping) -> str:
        """Returns the ground truth pathology for the given patient."""
        pathology = patient_entry["answer"]
        if pathology not in self.disease_mapping:
            raise ValueError(
                f"Rarebench: Could not find ground truth disease for patient pathology: {pathology}"
            )
        return self.disease_mapping[pathology]

    def get_patient_gt_ddx(self, patient_entry: Mapping) -> List[str]:
        """Returns the ground truth ddx for the given patient."""
        return []
