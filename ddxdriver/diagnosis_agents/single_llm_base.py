from abc import ABC, abstractmethod
from typing import final, Dict, List
import traceback 

from ddxdriver.models import init_model
from ddxdriver.utils import parse_differential_diagnosis, strip_all_lines, Patient, OutputDict, Constants
from ddxdriver.logger import log
from ddxdriver.benchmarks import Bench
from .utils import DiagnosisError
from .base import Diagnosis

class SingleLLMBase(Diagnosis):
    def __init__(self, diagnosis_agent_cfg):
        self.config = diagnosis_agent_cfg
        self.fewshot_cfg = self.config["fewshot"]

        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    @final
    def __call__(
        self,
        patient: Patient,
        bench: Bench,
        diagnosis_instructions: str = "",
        previous_pred_ddxs: List[List[str]] = [],
        previous_search_content: str = "",
    ) -> Dict:
        """
        Trying to diagnosis, loops through and retries
        This version uses a patient and assumes it has a patient profile
        """
        if not patient or not isinstance(patient.patient_profile, str) or not isinstance(patient.patient_initial_info, str):
            raise ValueError(
                "Calling diagnosis agent with None patient or patient profile/patient initial information as a non-empty string\n"
                f"Current patient or profile:\n{None if not patient else patient.patient_profile}"
            )
        # Resetting data used for diagnosis
        self.reset()
        retry_counter = 0
        while retry_counter < Constants.DIAGNOSIS_RETRIES.value:
            retry_counter += 1
            try:
                output_dict = self.diagnose(
                    patient=patient,
                    bench=bench,
                    diagnosis_instructions=diagnosis_instructions,
                    previous_pred_ddxs=previous_pred_ddxs,
                    previous_search_content=previous_search_content,
                )
                return output_dict
            except Exception as e:
                tb = traceback.format_exc()
                log.info(f"Caught error with diagnosis:\n {e}. Traceback:\n{tb}\nTrying again...")

        # If loop hasn't completed, then it errored more than we are specified to retry
        error_message = f"Did not diagnose in correct format in {Constants.DIAGNOSIS_RETRIES.value} tries, ending ddxdriver processing"
        raise Exception(error_message)

    @final
    def reset(self):
        """
        Resets per-patient data used to diagnose
        """
        self.message_history: Dict[str, str] = []
        self.parsed_ddx: List[str] = []
        self.last_diagnosis_error: DiagnosisError = None
