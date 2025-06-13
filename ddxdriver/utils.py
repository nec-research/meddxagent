import re
from typing import Dict, List, Tuple, Union, Any
from enum import Enum
import logging
from pathlib import Path

class Constants(Enum):
    SEED = 42
    DASH_NUMBER = 30
    AGENT_PROMPT_LENGTH = 10
    DIAGNOSIS_RETRIES = 5
    AGENT_CHOICE_RETRIES = 5
    RAG_RETRIES = 2


class Agents(Enum):
    HISTORY_TAKING = "history_taking"
    RAG = "rag"
    DIAGNOSIS = "diagnosis"


class OutputDict(Enum):
    """
    {
        PRED_DDX: List[str], ranked list of predicted diseases (i.e., 1. [DISEASE_1]\n 2. [DISEASE_2]\n)
        DDX_RATIONALE: the chain of thought rationale used to generate the ddx (Optional-for single_llm_cot)
        DIALOGUE_HISTORY: a DialogueHistory object
        RAG_CONTENT: a string representing the information you retrieved from the RAG class
        END: a boolean of whether to end DDxdriver steps
            - END is currently not used, it's created to allow future ddxdrivers which decide to end early
    }
    """

    PRED_DDX = "pred_ddx"
    DDX_RATIONALE = "ddx_rationale"
    DIALOGUE_HISTORY = "dialogue_history"
    RAG_CONTENT = "rag_content"
    END = "end"


def find_project_root(start_path=None) -> Path:
    """Find the project root directory. Adjust the method to suit your project structure."""
    if start_path is None:
        start_path = Path(__file__).resolve()

    # Look for a specific marker file or directory to determine the project root
    for parent in start_path.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    return start_path  # fallback to the script directory if marker not found


def strip_all_lines(s: str) -> str:
    """Remove all leading and trailing spaces of each line in the string."""
    return "\n".join(line.strip() for line in s.splitlines())


def parse_differential_diagnosis(diff_diag_str: str) -> List[str]:
    """
    Parses the first ranked list of differential diagnosis string into a list of diagnoses.
    Ignores any subsequent lists. Raises a ValueError if the first list doesn't contain
    valid ranked diagnoses.
    """
    # Regular expression to match the rank and the disease name
    pattern = re.compile(r"^\d+\.\s*(.+)", re.MULTILINE)
    
    # Find all matches in the entire string
    matches = pattern.findall(diff_diag_str)
    if not matches:
        raise ValueError("No valid ranked differential diagnosis found")

    # Find the first occurrence of a line starting with "1." to locate the start of the list
    lines = diff_diag_str.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(('1.', '1 .')):
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError("No valid ranked differential diagnosis found")

    # Only process lines from the first "1." until we hit a non-matching line
    final_matches = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if not pattern.match(line):
            break
        match = pattern.match(line)
        if match:
            final_matches.append(match.group(1))

    return final_matches


def ddx_list_to_string(diff_diag_list: List[str]) -> str:
    return "\n".join(f"{i + 1}. {pathology}" for i, pathology in enumerate(diff_diag_list))


class DialogueHistory:
    def __init__(self, input_dialogue: List[Tuple[str, str]] | None = None):
        # Dialogue history formatted as an ordered list dictionary of tuples, which can be either:
        # ("doctor", <doctor question>), or ("patient", <patient answer>)
        self.dialogue_history: List[Tuple[str, str]] = []
        if input_dialogue:
            self.dialogue_history.add_dialogue(dialogue=input_dialogue)

    def reset(self):
        self.dialogue_history = []

    def format_dialogue_history(self) -> str:
        conversation = ""
        for role, message in self.dialogue_history:
            if role == "doctor":
                conversation += f"Doctor: {message}\n"
            elif role == "patient":
                conversation += f"Patient: {message}\n"
            else:
                raise ValueError(f"Incorrect dialogue role of {role}")
        return conversation

    def add_dialogue(
        self,
        dialogue: Union[Tuple[str, str], List[Tuple[str, str]], "DialogueHistory"],
    ) -> None:
        # print(dialogue)
        # Ensure dialogue is a list of tuples
        if isinstance(dialogue, tuple):
            if len(dialogue) != 2 or not all(isinstance(x, str) for x in dialogue):
                raise TypeError("Expected a tuple of two strings")
            dialogue = [dialogue]
        elif isinstance(dialogue, list):
            if not all(
                isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, str) for x in item)
                for item in dialogue
            ):
                raise TypeError("Expected a list of tuples, each with exactly two strings")
        elif isinstance(dialogue, DialogueHistory):
            dialogue = dialogue.dialogue_history

        # Validate each dialogue entry
        for entry in dialogue:
            role, content = entry
            if role not in {"doctor", "patient"}:
                raise ValueError("The first element (role) must be either 'doctor' or 'patient'")
            self.dialogue_history.append(entry)


class Patient:

    def __init__(
        self,
        patient_id: Any = 0,
        patient_initial_info: str = "",
        gt_pathology: str = "",
        patient_profile: str = "",
        gt_ddx: List[str] = "",
    ):
        if (
            patient_id is None
            or not hasattr(patient_id, "__eq__")
            or not hasattr(patient_id, "__ne__")
        ):
            raise ValueError(
                "Error creating Patient, patient_id should be not None and have == / != defined"
            )
        self.patient_id = patient_id
        self.patient_initial_info = patient_initial_info
        self.gt_pathology = gt_pathology
        self.patient_profile = patient_profile
        self.gt_ddx = gt_ddx

    def __eq__(self, other):
        if not isinstance(other, Patient):
            return False
        return self.patient_id == other.patient_id

    def __ne__(self, other):
        # Use the __eq__ method to determine inequality
        return not self.__eq__(other)

    def pack_attributes(self) -> Dict:
        """
        Packs the patient attributes into a dictionary, which can be useful for json storage or using it's dynamic dictionary
        """
        return {
            "patient_id": self.patient_id,
            "patient_initial_info": self.patient_initial_info,
            "gt_pathology": self.gt_pathology,
            "patient_profile": self.patient_profile,
            "gt_ddx": self.gt_ddx,
        }

if __name__ == "__main__":
    print(find_project_root())