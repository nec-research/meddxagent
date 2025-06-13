from abc import ABC, abstractmethod
from typing import final
from copy import deepcopy

from ddxdriver.logger import log
from ddxdriver.utils import DialogueHistory, Patient


class PatientAgent(ABC):
    @abstractmethod
    def __init__(self, patient_agent_cfg):
        """
        Initialize the history taking agent (the doctor and the patient simulators)
        Uses DDxDriver to initialize each simulator with the input data
        """
        self.config = patient_agent_cfg
        self.patient: Patient = None
        self.dialogue_history = DialogueHistory()

    @final
    def reset(self, patient: Patient) -> None:
        """
        Resets patient and dialogue history with a new patient
        """
        # Deep copy so edits to the patient (i.e. updated patient profile) don't affect the patient agent
        self.patient = deepcopy(patient)
        self.dialogue_history.reset()

    @abstractmethod
    def __call__(self, question: str) -> str:
        """
        Calls history taking agent and initiates conversation
        Optionally, provides feedback to the doctor on how to proceed with the conversation
        """
