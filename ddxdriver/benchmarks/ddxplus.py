from ast import literal_eval
from typing import List, Mapping
import random
from datasets import Dataset, load_dataset
from ddxdriver.utils import Constants

from ddxdriver.utils import strip_all_lines, Patient

from .base import Bench


class DDxPlus(Bench):
    """..."""

    # Redefine variable of base class
    DATASET_NAME = "ddxplus"
    SPECIALIST_PREFACE = "You are a specialist in diseases where the chief complaint is related to cough, sore throat, or breathing issues."

    # Used to download the dataset
    DATASET_FILE = "appier-ai-research/StreamBench"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self) -> Dataset:
        """Loads the dataset."""
        dataset: Dataset = load_dataset(self.DATASET_FILE, self.DATASET_NAME)["test"]
        id_column = list(range(len(dataset)))
        dataset = dataset.add_column("id", id_column)
        return dataset

    def get_patient_profile(self, patient_entry: Mapping) -> str:
        """Returns the (full) patient profile."""
        return patient_entry["PATIENT_PROFILE"].split('""')[1]

    def get_patient_initial_info(self, patient_entry: Mapping) -> str:
        """Returns the initial patient information."""
        return strip_all_lines(
            f"""\
            Age: {patient_entry["AGE"]}
            Sex: {patient_entry["SEX"]}
            Chief Complaint: {patient_entry["INITIAL_EVIDENCE_ENG"]}"""
        )

    def get_patient_gt_pathology(self, patient_entry: Mapping) -> str:
        """Returns the ground truth pathology for the given patient."""
        return patient_entry["PATHOLOGY"]

    def get_patient_gt_ddx(self, patient_entry: Mapping) -> List[str]:
        """Returns the ground truth ddx for the given patient."""
        gt_ddx = literal_eval(patient_entry["DIFFERENTIAL_DIAGNOSIS"])
        gt_ddx = [item[0] for item in gt_ddx]
        return gt_ddx

    def get_fewshot_dataset(self) -> List[Patient]:
        """
        Returns a list of patients which can be sampled to find fewshot examples.

        By default, the entire patients list is used.
        """
        dataset: Dataset = load_dataset(self.DATASET_FILE, self.DATASET_NAME)["validate"]
        # Sample only 500 fewshot examples
        seed = Constants.SEED.value
        random.seed(seed)
        # Sample 500 indices from the dataset
        sample_indices = random.sample(range(len(dataset)), 500)
        # Select the samples using the indices
        dataset = dataset.select(sample_indices)

        id_column = [f"fewshot_{i}" for i in range(len(dataset))]
        dataset = dataset.add_column("id", id_column)
        try:
            patients = [
                self.create_patient(patient_entry=patient_entry) for patient_entry in dataset
            ]
            return patients
        except Exception as e:
            raise Exception(f"Error loading fewshot patients:\n{e}")
