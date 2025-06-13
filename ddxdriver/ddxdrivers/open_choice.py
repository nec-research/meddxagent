from typing import Dict

from ddxdriver.benchmarks import Bench
from ddxdriver.diagnosis_agents import Diagnosis
from ddxdriver.history_taking_agents import HistoryTaking
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.rag_agents import RAG
from ddxdriver.models import init_model
from ddxdriver.utils import DialogueHistory, strip_all_lines, OutputDict, Agents, Constants
from ddxdriver.logger import log

from .base import DDxDriver
from .utils import (
    dialogue_to_patient_profile,
    get_open_choice_system_prompt,
    get_open_choice_user_prompt,
    parse_open_choice_generation,
    get_fixed_choice_system_prompt,
    get_fixed_choice_user_prompt,
)


class OpenChoice(DDxDriver):
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
        ddxdriver_cfg["available_agents"] = ddxdriver_cfg.get("available_agents", valid_agents)

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

        # Rolling data over different calls
        self.new_dialogue_available = False
        self.agent_choice_history = []

    def step(self):
        "Calls whichever agent the turn is"

        # Validate that history taking is available if diagnosis is available and patient profile is None
        if (self.patient.patient_profile is None and 
            Agents.DIAGNOSIS.value in self.available_agents and 
            Agents.HISTORY_TAKING.value not in self.available_agents):
            raise ValueError("Error: History taking must be available when diagnosis is available and patient profile is None")

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

        if Agents.DIAGNOSIS.value in self.available_agents and self.turn_counter == self.max_turns:
            # Final turn, diagnose patient with available information
            log.info("On the final turn, calling diagnosis agent for a final diagnosis.")
            log.info("Generating agent prompt...")
            agent_choice = Agents.DIAGNOSIS.value
            system_prompt = get_fixed_choice_system_prompt(
                specialist_preface=self.bench.SPECIALIST_PREFACE
            )
            user_prompt = get_fixed_choice_user_prompt(
                agent_type=agent_choice,
                patient=self.patient,
                rag_content=self.get_final_rag_content(),
                previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                diagnosis_options=self.bench.diagnosis_options,
                agent_instruction_length=self.agent_prompt_length,
            )
            # log.info("User prompt:\n" + user_prompt + "\n\n")
            agent_prompt = self.model(user_prompt=user_prompt, system_prompt=system_prompt)
        else:
            # Choose agent and generate prompt
            log.info("Choosing agent + agent prompt...")
            retry_counter = 0
            choice_generated, correct_response = False, False
            agent_choice, agent_prompt = "", ""
            message_history = []
            
            while not correct_response and retry_counter < Constants.AGENT_CHOICE_RETRIES.value:
                retry_counter += 1
                if not choice_generated:
                    system_prompt = get_open_choice_system_prompt(
                        specialist_preface=self.bench.SPECIALIST_PREFACE,
                        max_turns=self.max_turns - 1,
                    )
                    # Having RAG agent prompt length be dependent on how many keyword searches
                    rag_search_length = (
                        self.rag_agent.max_keyword_searches
                        if self.rag_agent
                        else self.agent_prompt_length
                    )

                    user_prompt = get_open_choice_user_prompt(
                        turn_number=self.turn_counter,
                        available_agents=self.available_agents,
                        patient=self.patient,
                        rag_content=self.get_final_rag_content(),
                        previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                        diagnosis_options=self.bench.diagnosis_options,
                        agent_choice_history=self.agent_choice_history,
                        agent_instruction_length=self.agent_prompt_length,
                        rag_search_length=rag_search_length,
                    )
                    message_history = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    output = self.model(message_history=message_history)
                    choice_generated = True
                try:
                    agent_choice, agent_prompt = parse_open_choice_generation(text=output)
                    if ("[" in agent_choice) or ("]" in agent_choice):
                        agent_choice = agent_choice.replace("[", "").replace("]", "")
                        log.info(f"agent_choice: {agent_choice}")
                    if agent_choice not in self.available_agents:
                        log.info(
                            f"Error, invalid agent choice: {agent_choice} not in available agents: {self.available_agents}, trying again..."
                        )
                        message_history.extend([
                            {"role": "assistant", "content": output},
                            {"role": "user", "content": f"Your chosen agent was not among the available agents: {self.available_agents}. Please choose one agent from this list, given their specified function and the current information available. Just return the agent name for the agent choice, no additional text or brackets."}
                        ])
                        output = self.model(message_history=message_history)
                        continue
                    correct_response = True
                except Exception as e:
                    log.info(f"Caught error with open choice format:\n {e}, trying again...")
                    message_history.extend([
                        {"role": "assistant", "content": output},
                        {"role": "user", "content": f"Your previous output was in the incorrect format. Edit it to follow this exact format (replace the placeholders inside the brackets, and do not include the brackets themselves):\n[AGENT_CHOICE]\n{'-'*30}\n[AGENT_PROMPT]\n\nDirectly provide your response in the format specified, without additional text."}
                    ])
                    output = self.model(message_history=message_history)

            if not correct_response:
                # If loop hasn't completed, then it errored more than we are specified to retry
                error_message = f"Did not chose agent in correct format in {Constants.AGENT_CHOICE_RETRIES.value} tries, ending ddxdriver processing"
                raise Exception (error_message)

            log.info(f"Successfully chose {agent_choice} agent and generated agent prompt\n")
            self.agent_choice_history.append(agent_choice)

            # Optionally setting agent prompt to None if not RAG and specified
            if self.agent_prompt_length == 0 and agent_choice != Agents.RAG.value:
                agent_prompt = ""

        # Force history taking if patient profile is None and diagnosis was chosen
        if self.patient.patient_profile is None and agent_choice == Agents.DIAGNOSIS.value:
            log.info("Patient profile is None. Forcing history taking before diagnosis...")
            agent_choice = Agents.HISTORY_TAKING.value
            system_prompt = get_fixed_choice_system_prompt(
                specialist_preface=self.bench.SPECIALIST_PREFACE
            )
            user_prompt = get_fixed_choice_user_prompt(
                agent_type=agent_choice,
                patient=self.patient,
                rag_content=self.get_final_rag_content(),
                previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                diagnosis_options=self.bench.diagnosis_options,
                agent_instruction_length=self.agent_prompt_length,
            )
            agent_prompt = self.model(user_prompt=user_prompt, system_prompt=system_prompt)

        # Calling agent based on agent turn and agent prompt
        if agent_choice == Agents.HISTORY_TAKING.value:
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
        elif agent_choice == Agents.RAG.value:
            if not self.rag_agent:
                log.info("No rag agent defined, skipping rag...\n")
                return
            log.info("Starting RAG...\n")
            input_search = agent_prompt
            log.info("Input search for RAG:\n" + input_search + "\n")
            return self.rag_agent(
                input_search=input_search, diagnosis_options=self.bench.diagnosis_options
            )
        elif agent_choice == Agents.DIAGNOSIS.value:
            if not self.diagnosis_agent:
                log.info("No diagnosis agent defined, skipping diagnosis...")
                return
            log.info("Starting diagnosis...\n")
            diagnosis_instructions = agent_prompt
            log.info("Diagnosis Instructions:\n" + diagnosis_instructions + "\n")
            if self.agent_choice_history and self.agent_choice_history[-1] == Agents.RAG.value:
                previous_search_content = self.get_final_rag_content()
            else:
                previous_search_content = ""
            output = self.diagnosis_agent(
                patient=self.patient,
                bench=self.bench,
                diagnosis_instructions=diagnosis_instructions,
                previous_pred_ddxs=self.get_k_final_ddxs(k=self.K_PREVIOUS_DDXS),
                previous_search_content=self.get_final_rag_content(),
            )
            return output
        else:
            error_message = f"Error, invalid agent choice: {agent_choice}, returning early from ddxdriver..."
            raise Exception(error_message)
