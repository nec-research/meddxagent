from typing import Dict, List

from ddxdriver.models import init_model

from ddxdriver.utils import (
    parse_differential_diagnosis,
    strip_all_lines,
    Patient,
    OutputDict,
    Constants,
)
from ddxdriver.benchmarks import Bench

from .single_llm_base import SingleLLMBase
from .utils import (
    get_ddx_system_prompt,
    get_ddx_user_prompt,
    get_self_generate_ddx_cot_user_prompt,
    DiagnosisError,
)
from ddxdriver.logger import log


class SingleLLMStandard(SingleLLMBase):
    def __init__(self, diagnosis_agent_cfg):
        super().__init__(diagnosis_agent_cfg)

    def diagnose(
        self,
        patient: Patient,
        bench: Bench,
        diagnosis_instructions: str = "",
        previous_pred_ddxs: List[List[str]] = [],
        previous_search_content: str = "",
    ) -> Dict:

        output = ""
        user_prompt = ""
        # First time running through
        if not self.message_history:
            # Error checking because of complex prompting
            try:
                system_prompt = get_ddx_system_prompt(
                    diagnosis_options=bench.diagnosis_options,
                    specialist_preface=bench.SPECIALIST_PREFACE,
                    ddx_length=bench.DDX_LENGTH,
                    use_cot=False,
                )
                # log.info("System prompt:\n\n" + system_prompt + "\n\n")
                # exit()
                shots = bench.get_fewshot(
                    patient=patient,
                    fewshot_cfg=self.fewshot_cfg,
                )
                shot_dicts = [patient.pack_attributes() for patient in shots]
                user_prompt = get_ddx_user_prompt(
                    patient_profile=patient.patient_profile,
                    shot_dicts=shot_dicts,
                    use_cot=False,
                    diagnosis_instructions=diagnosis_instructions,
                    previous_pred_ddxs=previous_pred_ddxs,
                    previous_search_content=previous_search_content,
                )
                log.info("\nDiagnosis User prompt:\n\n" + user_prompt + "\n")
                # exit()

            except Exception as e:
                error_message = f"Error with diagnosis prompting:\n{e}\nReturning early..."
                raise Exception(error_message)
            # Creating output and logging to message history
            output = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
            log.info("FIRST DIAGNOSIS Output:\n" + output + "\n")
            # exit()
            self.message_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output},
            ]
        else:
            match self.last_diagnosis_error:
                case DiagnosisError.DDX_FORMAT:
                    log.info("DIAGNOSIS ERROR: DDX FORMAT")
                    user_prompt = strip_all_lines(
                        f"""\
                        Your ranked differential diagnosis could not be correctly parsed as a numbered list of diagnoses. 
                        Please edit your ranked differential diagnosis to follow the correct format, with one diagnosis per line (replace the placeholders inside the brackets, and do not include the brackets themselves): [RANK_NUMBER]. [DIAGNOSIS]. 
                        I.e.:
                        1. [DIAGNOSIS_1]
                        2. [DIAGNOSIS_2]
                        ...
                        Still maintain the format (do not include the angle brackets):
                        [STEP-BY-STEP_RATIONALE]
                        {'-'*Constants.DASH_NUMBER.value}
                        Ranked Differential Diagnosis:
                        [DDX]

                        Directly provide your response in the format specified, without additional text.
                        """
                    )
                case DiagnosisError.DDX_LENGTH:
                    log.info("DIAGNOSIS ERROR: DDX LENGTH")
                    user_prompt = strip_all_lines(
                        f"""\
                        Your ranked differential diagnosis contained {len(self.parsed_ddx)} diseases, when it should have had {bench.DDX_LENGTH}.
                        Please consider the conversation history and edit your output to return a ranked differential diagnosis of length {bench.DDX_LENGTH}.
                        You should still maintain the format (replace the placeholders inside the brackets, and do not include the brackets themselves):
                        [STEP-BY-STEP_RATIONALE]
                        {'-'*Constants.DASH_NUMBER.value}
                        Ranked Differential Diagnosis:
                        [DDX]

                        Directly provide your response in the format specified, without additional text.
                        """
                    )
            # log.info("USER PROMPT:\n" + user_prompt + "\n")
            # log.info("MESSAGE HISTORY:\n" + str(self.message_history) + "\n")
            # Running output with updated message history + error user prompt
            output = self.model(user_prompt=user_prompt, message_history=self.message_history)
            # log.info("FIX DIAGNOSIS OUTPUT:\n" + output + "\n")
        # Checking for errors in output
        try:
            self.parsed_ddx = parse_differential_diagnosis(diff_diag_str=output)
        except Exception as e:
            log.info("DIAGNOSIS ERROR: DDX FORMAT")
            self.last_diagnosis_error = DiagnosisError.DDX_FORMAT
            raise Exception(f"Caught error trying to parse differential diagnosis\n{e}")
        if not self.parsed_ddx or (bench.DDX_LENGTH and len(self.parsed_ddx) != bench.DDX_LENGTH):
            log.info("LENGTH OF PREDICTED DDX IS INCORRECT\n")
            self.last_diagnosis_error = DiagnosisError.DDX_LENGTH
            raise ValueError(
                f"Length of predicted ddx is incorrect, it is either 0 or does not match the bench's required length of {bench.DDX_LENGTH}"
            )

        # All errors checked, correctly formatted
        return {OutputDict.PRED_DDX: self.parsed_ddx}
