from abc import ABC, abstractmethod
from typing import final, Dict, Any, List

from ddxdriver.benchmarks import Bench
from ddxdriver.diagnosis_agents import Diagnosis
from ddxdriver.history_taking_agents import HistoryTaking
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.rag_agents import RAG
from ddxdriver.utils import (
    DialogueHistory,
    Patient,
    OutputDict,
    Agents,
    ddx_list_to_string,
    Constants,
)
from ddxdriver.logger import log


"The ddxdriver, which can be a simple llm, or an autogen with conversable agent wrappers which call code"


class DDxDriver(ABC):
    """
    Base class for ddxdriver. Defines methods for loading general config information, running the ddxdriver, and managing data
    """

    MAX_POSSIBLE_TURNS = 15
    DEFAULT_MAX_TURNS = 3
    K_PREVIOUS_DDXS = 2

    def __init__(
        self,
        ddxdriver_cfg,
        bench: Bench | None = None,
        diagnosis_agent: Diagnosis | None = None,
        history_taking_agent: HistoryTaking | None = None,
        patient_agent: PatientAgent | None = None,
        rag_agent: RAG | None = None,
    ):
        self.config = ddxdriver_cfg

        # Patient data
        self.bench = bench
        self.dialogue_history = DialogueHistory()

        # Initializing all agents (diagnosis, history_taking, rag) and error checking.
        self.diagnosis_agent = diagnosis_agent
        self.history_taking_agent = history_taking_agent
        self.patient_agent = patient_agent
        self.rag_agent = rag_agent

        # Can specify which agents are avilable with the available_agents key in ddxdriver config, defaults to all available
        self.init_available_agents()

        # Turn information used in self.call() loop
        self.max_turns = self.config.get("max_turns")
        if not isinstance(self.max_turns, int) or self.max_turns > self.MAX_POSSIBLE_TURNS:
            log.warning(
                f"Warning, max turns is unspecified, invalid, or about the max of {self.MAX_POSSIBLE_TURNS}. Using {self.DEFAULT_MAX_TURNS} max turns..."
            )
            self.max_turns = self.DEFAULT_MAX_TURNS
        self.turn_counter = 1

        # Other data
        self.agent_prompt_length: int = self.config.get(
            "agent_prompt_length", Constants.AGENT_PROMPT_LENGTH.value
        )
        self.only_patient_initial_information = self.config.get("only_patient_initial_information", False)
    
    @final
    def __call__(self, patient: Patient):
        "Calls ddxdriver to run a complete example for the patient"
        if patient is None:
            raise ValueError("Trying to call ddxdriver with patient as None")
        self.reset(patient)
        self.turn_counter = 1
        while not self.max_turns or self.turn_counter <= self.max_turns:
            log.info(f"\nSTEP...Calling ddxdriver.step() for turn number {self.turn_counter}...\n\n")
            try:
                output_dict = self.step()
            except Exception as e:
                raise Exception(f"Exception in DDxDriver.step():\n{e}")
            # If a agent was called and returned an output_dict, increment turns
            if output_dict:
                self.turn_counter += 1
            self.parse_output_dict(output_dict)
            if output_dict and output_dict.get(OutputDict.END, False):
                break

    @abstractmethod
    def step(self) -> Dict:
        """
        Tells ddxdriver to make next move for a given patient, returning an output dictionary with keys in OutputDict (in ddxdriver.utils):
        "OutputDict.PRED_DDX": List[str], ranked list of predicted diseases (i.e., 1. [DISEASE_1]\n 2. [DISEASE_2]\n)
        "OutputDict.DDX_RATIONALE": the chain of thought rationale used to generate the ddx (Optional-for single_llm_cot)
        "OutputDict.DIALOGUE_HISTORY": a DialogueHistory object
        "OutputDict.RAG_CONTENT": a string representing the information you retrieved from the RAG class
        "OutputDICT.END": a boolean of whether to end DDxdriver steps
        }
        Can raise an Exception given an error
        """
        raise NotImplemented

    @final
    def reset(self, patient: Patient) -> None:
        "Resets the dialogue history, patient, and rolling data with a new patient"
        if not patient:
            raise ValueError("Trying to reset DDxDriver with patient, but patient is None")
        # Rolling data
        self.pred_ddxs, self.pred_ddx_rationales, self.rag_content = [], [], []
        self.patient = patient
        self.dialogue_history.reset()
        # Reinitializes patient agent with patient (deep copy)
        if self.patient_agent:
            self.patient_agent.reset(patient)
        if self.only_patient_initial_information:
            log.info("Setting patient profile to patient initial information because only_patient_initial_information is True")
            log.info(f"Patient initial information:\n {self.patient.patient_initial_info}")
            self.patient.patient_profile = self.patient.patient_initial_info
        if self.history_taking_agent:
            self.history_taking_agent.dialogue_history.reset()
            # Sets DDxdriver's patient profile to None (no initial patient profile if history taking used)
            log.info(
                "Setting patient profile to None because history taking agent is available...\n"
            )
            self.patient.patient_profile = None

    @final
    def add_ddx(self, pred_ddx: List[str]):
        if not isinstance(pred_ddx, list) or not all(isinstance(item, str) for item in pred_ddx):
            raise TypeError("expected a List[str] as input")
        self.pred_ddxs.append(pred_ddx)

    @final
    def add_ddx_rationale(self, pred_ddx_rationale: str):
        if not isinstance(pred_ddx_rationale, str):
            raise TypeError("expected a str as input")
        self.pred_ddx_rationales.append(pred_ddx_rationale)

    @final
    def add_rag_content(self, rag_content: str):
        if not isinstance(rag_content, str):
            raise TypeError("expected a str as input")
        self.rag_content.append(rag_content)

    @final
    def add_dialogue_history(self, dialogue: DialogueHistory) -> None:
        "Adds dialogue_history (type checking is done within the DialogueHistory class)"
        if not isinstance(dialogue, DialogueHistory):
            raise TypeError("Expected DialogueHistory object")
        self.dialogue_history.add_dialogue(dialogue)

    @final
    def get_dialogue_history(self) -> str:
        "Returns string of DialogueHistory"
        return self.dialogue_history.format_dialogue_history() if self.dialogue_history else ""

    @final
    def get_final_ddx(self) -> List[str]:
        if len(self.pred_ddxs) == 0:
            return []
        final_ddx = self.pred_ddxs[-1]
        if not isinstance(final_ddx, list) or not all(isinstance(x, str) for x in final_ddx):
            log.error("Error, final ddx should be a List[str], returning an empty list...")
            return []
        return final_ddx

    @final
    def get_final_ddx_rationale(self) -> str:
        if len(self.pred_ddx_rationales) == 0:
            return ""
        final_ddx_rationale = self.pred_ddx_rationales[-1]
        if not isinstance(final_ddx_rationale, str):
            log.error("Error, final ddx rationale should be a string, returning an empty string...")
            return ""
        return final_ddx_rationale

    @final
    def get_final_rag_content(self) -> str:
        if len(self.rag_content) == 0:
            return ""
        final_rag_content = self.rag_content[-1]
        if not isinstance(final_rag_content, str):
            log.error("Error, final rag content should be a string, returning an empty string...")
            return ""
        return final_rag_content

    @final
    def get_k_final_ddxs(self, k: int) -> List[List[str]]:
        if len(self.pred_ddxs) == 0 or k <= 0:
            return []
        final_ddx = self.pred_ddxs[-k:]
        if not all(
            (isinstance(ddx, list) and all(isinstance(x, str) for x in ddx)) for ddx in final_ddx
        ):
            log.error("Error, final_ddxs should be a List[List[str]], returning an empty list...")
            return []
        return final_ddx

    @final
    def init_available_agents(self):
        """
        Given the available_agents parameter in config, ensures that these components are available or set to None
        The available_agents is a subset of ["diagnosis", "history_taking", "rag"] (agents enum in utils.py)
        For example, if available_agents is ["diagnosis", "rag"], it will:
            Ensure diagnosis and rag agents are available (not None)
            Set history_taking to None
        Will also ensure that if history_taking is available, both history_taking agent and a patient agent is available (for conversation)
        """
        # Initialize available agents
        valid_agents = [agent.value for agent in Agents]
        self.available_agents = self.config.get("available_agents", valid_agents)
        if not isinstance(self.available_agents, list) or not all(
            agent in valid_agents for agent in self.available_agents
        ):
            raise ValueError(
                "Error initializing DDxDriver: Available agents has an incorrect agent name\n"
                f"Should be subset of: {valid_agents}"
            )

        # Ensure that available agents are truly available or set to none
        if Agents.DIAGNOSIS.value in self.available_agents:
            if not self.diagnosis_agent:
                raise ValueError(
                    f"Error initializing DDxDriver: {Agents.DIAGNOSIS.value} in available agents but diagnosis_agent is None"
                )
        else:
            self.diagnosis_agent = None
        if Agents.HISTORY_TAKING.value in self.available_agents:
            if not self.history_taking_agent:
                raise ValueError(
                    f"Error initializing DDxDriver: {Agents.HISTORY_TAKING.value} in available agents but history_taking_agent is None"
                )
        else:
            self.history_taking_agent = None
        if Agents.RAG.value in self.available_agents:
            if not self.rag_agent:
                raise ValueError(
                    f"Error initializing DDxDriver: {Agents.RAG.value} in available agents but rag_agent is None"
                )
        else:
            self.rag_agent = None

        # Ensuring that if history_taking defined, then a patient agent is as well.
        if self.history_taking_agent and not self.patient_agent:
            raise ValueError(
                "Error initializing DDxDriver: If using history_taking agent, then a patient_agent must also be not None "
                "(in addition to history_taking agent)"
            )

    @final
    def parse_output_dict(self, output_dict: Dict[str, Any]):
        """
        Parameters:
            output_dict: Dictionary with keys in OutputDict (in ddxdriver.utils)
        Output:
            Adds dictionary values' (if non-empty) to ddxdriver data
        """
        if not output_dict:
            return
        if OutputDict.PRED_DDX in output_dict:
            log.info(
                f"\nAdding ddx to ddxdriver.pred_ddxs (ground truth pathology: {self.patient.gt_pathology}):\n"
                + ddx_list_to_string(output_dict[OutputDict.PRED_DDX])
                + "\n"
            )
            self.add_ddx(output_dict[OutputDict.PRED_DDX])
        if OutputDict.DDX_RATIONALE in output_dict:
            self.add_ddx_rationale(output_dict[OutputDict.DDX_RATIONALE])
        if OutputDict.DIALOGUE_HISTORY in output_dict:
            self.add_dialogue_history(output_dict[OutputDict.DIALOGUE_HISTORY])
        if OutputDict.RAG_CONTENT in output_dict:
            self.add_rag_content(output_dict[OutputDict.RAG_CONTENT])
