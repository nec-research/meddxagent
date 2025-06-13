from typing import Dict

from ddxdriver.models import init_model
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.benchmarks import Bench
from ddxdriver.utils import OutputDict

from .base import HistoryTaking
from .utils import get_history_taking_system_prompt, get_history_taking_user_prompt
from ddxdriver.logger import log

class LLMHistoryTaking(HistoryTaking):
    def __init__(self, history_taking_agent_cfg):
        super().__init__(history_taking_agent_cfg)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    def __call__(
        self,
        patient_agent: PatientAgent,
        bench: Bench,
        conversation_goals: str = "",
    ) -> Dict:
        specialist_preface = bench.SPECIALIST_PREFACE
        num_questions = 0
        while num_questions < self.max_questions:
            system_prompt = get_history_taking_system_prompt(
                specialist_preface=specialist_preface,
            )
            # log.info(system_prompt + "\n\n")

            user_prompt = get_history_taking_user_prompt(
                patient_initial_info=patient_agent.patient.patient_initial_info,
                dialogue_history_text=self.dialogue_history.format_dialogue_history(),
                conversation_goals=conversation_goals,
            )
            # log.info("User prompt:\n\n" + user_prompt + "\n\n")
            # exit()
            # import time

            # start = time.time()
            # log.info("Starting to generate question...")
            question = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
            # log.info("Time taken to gen question: ", time.time() - start)
            log.info("Doctor: " + question + "\n")
            self.dialogue_history.add_dialogue(("doctor", question))
            if question == "None":
                break
            answer = patient_agent(question=question)
            self.dialogue_history.add_dialogue(("patient", answer))
            log.info("Patient: " + answer + "\n")
            num_questions += 1
        return {OutputDict.DIALOGUE_HISTORY: self.dialogue_history}
