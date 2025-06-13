from abc import ABC, abstractmethod
from typing import Dict

from ddxdriver.utils import DialogueHistory
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.benchmarks import Bench

class HistoryTaking(ABC):
    @abstractmethod
    def __init__(self, history_taking_agent_cfg):
        """
        Initialize the history taking agent (the agent in charge of talking with the patient)
        Uses DDxDriver to initialize each simulator with the input data
        """
        self.config = history_taking_agent_cfg
        self.max_questions = self.config["max_questions"]
        self.dialogue_history = DialogueHistory()

    @abstractmethod
    def __call__(
        self,
        patient_agent: PatientAgent,
        bench: Bench,
        conversation_goals: str = "",
    ) -> Dict:
        """
        Calls history taking agent and initiates conversation with patient
        Optionally, integrates conversation goals from ddxdriver on how to proceed with the conversation
        """
