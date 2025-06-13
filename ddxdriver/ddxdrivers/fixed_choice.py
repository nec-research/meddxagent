from typing import Dict

from ddxdriver.benchmarks import Bench
from ddxdriver.diagnosis_agents import Diagnosis
from ddxdriver.history_taking_agents import HistoryTaking
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.rag_agents import RAG
from ddxdriver.models import init_model
from ddxdriver.utils import Agents, Patient
from ddxdriver.logger import log

from .base import DDxDriver
from .utils import (
    dialogue_to_patient_profile,
    get_fixed_choice_system_prompt,
    get_fixed_choice_user_prompt,
)


class FixedChoice(DDxDriver):
    """
    A baseline ddxdriver which does a single iteration of calling modular components
    Used to validate/test system or perform ablation studies
    """

    def __init__(
        self,
        ddxdriver_cfg,
        bench: Bench | None = None,
        diagnosis_agent: Diagnosis | None = None,
        history_taking_agent: HistoryTaking | None = None,
        patient_agent: PatientAgent | None = None,
        rag_agent: RAG | None = None,
    ):
        # Making available agents those specified in the agent_order
        valid_agents = [agent.value for agent in Agents]
        self.agent_order = ddxdriver_cfg.get("agent_order", valid_agents)
        ddxdriver_cfg["available_agents"] = self.agent_order
        # Setting max turns to iteration count * length of agent_order
        iterations = ddxdriver_cfg.get("iterations", 1)
        ddxdriver_cfg["max_turns"] = iterations * len(self.agent_order)

        super().__init__(
            bench=bench,
            ddxdriver_cfg=ddxdriver_cfg,
            diagnosis_agent=diagnosis_agent,
            history_taking_agent=history_taking_agent,
            patient_agent=patient_agent,
            rag_agent=rag_agent,
        )

        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

        self._reset_agent_order()

    def _reset_agent_order(self):
        """Reset the agent order index and related state variables."""
        self.agent_order_idx = 0
        self.new_dialogue_available = False
        self.previous_agent = ""

    def reset(self, patient: Patient) -> None:
        """Override parent reset to also reset agent order."""
        super().reset(patient)
        self._reset_agent_order()

    def get_agent_turn(self):
        """
        Looks at self.agent_order (a list of agents and the order to call them),
        and sets self.agent_order_idx for the current one to call
        """
        agent_turn = self.agent_order[self.agent_order_idx]
        self.agent_order_idx = (self.agent_order_idx + 1) % len(self.agent_order)
        return agent_turn

    def step(self):
        "Calls whichever agent the turn is"

        # Generate and update current_patient_profile if new dialogue history. Doing this here since dialogue will have been logged to ddxdriver
        if self.new_dialogue_available and (dialogue_history_text := self.get_dialogue_history()):
            log.info("Updating Patient Profile...\n")
            updated_patient_profile = dialogue_to_patient_profile(
                dialogue_history_text=dialogue_history_text,
                patient=self.patient,
                model=self.model,
            )
            if updated_patient_profile:
                self.patient.patient_profile = updated_patient_profile
                log.info(
                    "Updated Patient Profile:\n" + self.patient.patient_profile + "\n",
                )
                self.new_dialogue_available = False

        # Generate new turn
        agent_turn = self.get_agent_turn()
        self.previous_agent = agent_turn
        # If this example will end the next turn, resetting the turn index to 0
        if self.turn_counter == self.max_turns:
            self.agent_order_idx = 0
            self.previous_agent = ""

        if self.agent_prompt_length == 0 and agent_turn != Agents.RAG.value:
            # Optionally setting agent prompt to None if not RAG and specified
            agent_prompt = ""
        else:
            # Generate new agent prompt
            log.info("Generating agent prompt...")
            system_prompt = get_fixed_choice_system_prompt(
                specialist_preface=self.bench.SPECIALIST_PREFACE
            )

            # Having RAG agent prompt length be dependent on how many keyword searches
            rag_search_length = (
                self.rag_agent.max_keyword_searches if self.rag_agent else self.agent_prompt_length
            )

            user_prompt = get_fixed_choice_user_prompt(
                agent_type=agent_turn,
                patient=self.patient,
                rag_content=self.get_final_rag_content(),
                previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                diagnosis_options=self.bench.diagnosis_options,
                agent_instruction_length=self.agent_prompt_length,
                rag_search_length=rag_search_length,
            )
            # log.info("Fixed Choice DDxDriver User prompt:\n" + user_prompt + "\n\n")
            agent_prompt = self.model(user_prompt=user_prompt, system_prompt=system_prompt)

        # Calling agent based on agent turn and agent prompt
        if agent_turn == Agents.HISTORY_TAKING.value:
            if not self.history_taking_agent:
                log.info("No history taking agent defined, skipping history taking...\n")
                return
            log.info("Starting history taking...\n")
            self.new_dialogue_available = True
            conversation_goals = agent_prompt
            log.info(f"Conversation goals for history taking:\n" + conversation_goals + "\n")
            return self.history_taking_agent(
                patient_agent=self.patient_agent,
                bench=self.bench,
                conversation_goals=conversation_goals,
            )
        elif agent_turn == Agents.RAG.value:
            if not self.rag_agent:
                log.info("No rag agent defined, skipping rag...\n")
                return
            log.info("Starting RAG...\n")
            input_search = agent_prompt
            log.info("Input search for RAG:\n" + input_search + "\n")
            return self.rag_agent(
                input_search=input_search, diagnosis_options=self.bench.diagnosis_options
            )
        elif agent_turn == Agents.DIAGNOSIS.value:
            if not self.diagnosis_agent:
                log.info("No diagnosis agent defined, skipping diagnosis...")
                return
            log.info("Starting diagnosis...\n")
            diagnosis_instructions = agent_prompt
            log.info("Diagnosis Instructions:\n" + diagnosis_instructions + "\n")
            previous_search_content = (
                self.get_final_rag_content() if self.previous_agent == Agents.RAG.value else ""
            )
            return self.diagnosis_agent(
                patient=self.patient,
                bench=self.bench,
                diagnosis_instructions=diagnosis_instructions,
                previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                previous_search_content=previous_search_content,
            )
        else:
            error_message= "Invalid turn number, returning early from ddxdriver..."
            raise Exception(error_message)
