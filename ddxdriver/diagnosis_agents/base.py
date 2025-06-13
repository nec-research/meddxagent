from abc import ABC, abstractmethod
from typing import Dict, List

from ddxdriver.utils import Patient
from ddxdriver.benchmarks import Bench


class Diagnosis(ABC):
    @abstractmethod
    def __init__(self, diagnosis_agent_cfg):
        "Initialize the diagnosis agent"
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        patient: Patient,
        bench: Bench,
        diagnosis_instructions: str = "",
        previous_pred_ddxs: List[List[str]] = [],
        previous_search_content: str = "",
    ) -> Dict:
        """
        A loop which calls diagnose function, which returns diagnosis_dict from diagnose() function to the ddxdriver
        """
        raise NotImplementedError

    @abstractmethod
    def diagnose(
        self,
        patient: Patient,
        bench: Bench,
        diagnosis_instructions: str = "",
        previous_pred_ddxs: List[List[str]] = [],
        previous_search_content: str = "",
    ) -> Dict:
        """
        Returns a dictionary of diagnosis predictions info, or raises an error
        Paramters:
            patient: instance of patient class
            bench: instance of bench class
            diagnosis_instructions (optional): diagnosis instructions to guide the ddx
            previous_pred_ddx (optional): the previous ranked differential diagnosis that was predicted
        Output
        Dictionary of diagnosis information, which can look like
        {
            OutputDict.PRED_DDX: List[str], ranked list of predicted diseases (i.e., 1. <diease>\n 2. <disease>\n)
            OutputDict.DDX_RATIONALE: str, the chain of thought rationale used to generate the ddx (Optional-for single_llm_cot)
        }
        """
        raise NotImplementedError

    def reset(self):
        """
        Optional method to reset diagnosis agent data
        """
