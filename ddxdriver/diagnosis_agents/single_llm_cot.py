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
    parse_cot_generation,
    get_ddx_system_prompt,
    get_ddx_user_prompt,
    get_self_generate_ddx_cot_user_prompt,
    DiagnosisError,
)
from ddxdriver.logger import log


class SingleLLMCOT(SingleLLMBase):
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
                    use_cot=True,
                )
                # log.info("System prompt:\n\n" + system_prompt + "\n\n")
                # exit()
                shots = bench.get_fewshot(
                    patient=patient,
                    fewshot_cfg=self.fewshot_cfg,
                )
                shot_dicts = [patient.pack_attributes() for patient in shots]
                if self.fewshot_cfg.get("self_generated_fewshot_cot", False):
                    for shot_dict in shot_dicts:
                        user_prompt = get_self_generate_ddx_cot_user_prompt(
                            shot_dict=shot_dict,
                            diagnosis_options=bench.diagnosis_options,
                            specialist_preface=bench.SPECIALIST_PREFACE,
                            ddx_length=bench.DDX_LENGTH,
                        )
                        shot_dict["cot_rationale"] = self.model(user_prompt=user_prompt)

                user_prompt = get_ddx_user_prompt(
                    patient_profile=patient.patient_profile,
                    shot_dicts=shot_dicts,
                    use_cot=True,
                    diagnosis_instructions=diagnosis_instructions,
                    previous_pred_ddxs=previous_pred_ddxs,
                    previous_search_content=previous_search_content,
                )
                log.info("\nDiagnosis User prompt:\n\n" + user_prompt + "\n")
                # exit()
            except Exception as e:
                error_message = f"Error with creating COT diagnosis prompting:\n{e}\nReturning early..."
                raise Exception(error_message)

            # Creating output and logging to message history
            output = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
            self.message_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output},
            ]
        else:
            match self.last_diagnosis_error:
                case DiagnosisError.COT_FORMAT:
                    user_prompt = strip_all_lines(
                        f"""\
                        Your previous output was in the incorrect format. Edit it to follow this exact format (replace the placeholders inside the brackets, and do not include the brackets themselves):
                        [STEP-BY-STEP_RATIONALE]
                        {'-'*Constants.DASH_NUMBER.value}
                        Ranked Differential Diagnosis:
                        [DDX]

                        Directly provide your response in the format specified, without additional text.
                        """
                    )
                case DiagnosisError.DDX_FORMAT:
                    user_prompt = strip_all_lines(
                        f"""\
                        Your ranked differential diagnosis could not be correctly parsed as a numbered list of diagnoses. 
                        Please edit your ranked differential diagnosis to follow the correct format, with one diagnosis per line (replace the placeholders inside the brackets, and do not include the brackets themselves): [RANK_NUMBER]. [DIAGNOSIS]. I.e.:
                        1. [DIAGNOSIS_1]
                        2. [DIAGNOSIS_2]
                        ...
                        Maintain this response format (replace the placeholders inside the brackets, and do not include the brackets themselves):
                        [STEP-BY-STEP_RATIONALE]
                        {'-'*Constants.DASH_NUMBER.value}
                        Ranked Differential Diagnosis:
                        [DDX]

                        Directly provide your response in the format specified, without additional text.
                        """
                    )
                case DiagnosisError.DDX_LENGTH:
                    user_prompt = strip_all_lines(
                        f"""\
                        Your ranked differential diagnosis contained {len(self.parsed_ddx)} diseases, when it should have had {bench.DDX_LENGTH}.
                        Please consider the conversation history and edit your output to return a ranked differential diagnosis of length {bench.DDX_LENGTH}.
                        Maintain this response format (replace the placeholders inside the brackets, and do not include the brackets themselves):
                        [STEP-BY-STEP_RATIONALE]
                        {'-'*Constants.DASH_NUMBER.value}
                        Ranked Differential Diagnosis:
                        [DDX]

                        Directly provide your response in the format specified, without additional text.
                        """
                    )

            # Running output with updated message history + error user prompt
            output = self.model(user_prompt=user_prompt, message_history=self.message_history)

        # Checking for errors in output
        try:
            pred_ddx, ddx_rationale = parse_cot_generation(text=output)
        except Exception as e:
            self.last_diagnosis_error = DiagnosisError.COT_FORMAT
            raise Exception(f"COT format is incorrect, caught exception:\n{e}")
        try:
            self.parsed_ddx = parse_differential_diagnosis(diff_diag_str=pred_ddx)
        except Exception as e:
            self.last_diagnosis_error = DiagnosisError.DDX_FORMAT
            raise Exception(f"Caught error trying to parse differential diagnosis\n{e}")
        if not self.parsed_ddx or (bench.DDX_LENGTH and len(self.parsed_ddx) != bench.DDX_LENGTH):
            self.last_diagnosis_error = DiagnosisError.DDX_LENGTH
            raise ValueError(
                f"Length of predicted ddx is incorrect, it is either 0 or does not match the bench's required length of {bench.DDX_LENGTH}"
            )

        # All errors checked, correctly formatted
        return {OutputDict.PRED_DDX: self.parsed_ddx, OutputDict.DDX_RATIONALE: ddx_rationale}
