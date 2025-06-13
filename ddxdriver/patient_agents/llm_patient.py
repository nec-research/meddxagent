from ddxdriver.models import init_model

from .utils import get_patient_system_prompt, get_patient_user_prompt

from .base import PatientAgent


class LLMPatient(PatientAgent):
    def __init__(self, patient_agent_cfg):
        super().__init__(patient_agent_cfg)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    def __call__(self, question: str) -> str:
        self.dialogue_history.add_dialogue(("doctor", question))

        system_prompt = get_patient_system_prompt(
            patient_profile=self.patient.patient_profile,
        )
        user_prompt = get_patient_user_prompt(
            patient_initial_info=self.patient.patient_initial_info,
            dialogue_history_text=self.dialogue_history.format_dialogue_history(),
        )
        answer = self.model(user_prompt=user_prompt, system_prompt=system_prompt)

        self.dialogue_history.add_dialogue(("patient", answer))
        return answer
