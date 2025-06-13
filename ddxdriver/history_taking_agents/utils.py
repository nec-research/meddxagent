from typing import List

from ddxdriver.utils import strip_all_lines


def get_history_taking_system_prompt(
    specialist_preface: str | None = None,
):
    specialist_preface_prompt = (
        specialist_preface if specialist_preface else "You are a medical doctor."
    )
    prompt = f"""\
        {specialist_preface_prompt}
        Your job is to take medical history from a patient by asking them specific questions to determine their antecedents and symptoms, as well as narrow down the possible diseases they may be suffering from. This information will assist in creating a differential diagnosis of the patient's most likely diseases. You may end the conversation by responding with the word \'None\'.

        You may receive this additional information to guide your dialogue:
        - Initial Patient Information: Information the patient has already self-reported, such as chief complaint, age, sex, etc.
        - Dialogue History: The conversation you and the patient have had so far, formatted as \'Doctor\' / \'Patient\' turns.
        - Suggested Conversation Goals: Specific topics or questions to try to cover in the dialogue. You may also ask questions outside of these conversation goals; do not limit yourself to these.

        You may either start, end, or continue the conversation, as explained below:
        1) Start the conversation: If the Dialogue History is empty, start the conversation by asking a question to answer about their antecedents/symptoms.
        2) End the conversation: If you believe you have enough information or the patient doesn't know anything else, end the conversation by responding with the word \'None\'
        3) Continue the conversation: If you believe you can learn more relevant patient history and narrow down more possible diseases, ask the patient another question about their antecedents/symptoms.

        Response Instructions:
        - Respond with only one question about a single topic, including antecedents/symptoms
        - At the beginning of the dialogue, you should start off asking the patient questions which determine what area of diseases they likely have. You may ask for what kind of symptoms or antecedents/diseases they mainly have. If you have a medical specialty, focus your questions in that domain.
        - As the dialogue progresses, you should narrow your questions to specific types of antecedents/symptoms which you believe are relevant to the diseases the patient likely has    
        - You may clarify chief complaints/main symptoms.
        - If Conversation Goals are specified, focus your questions on accomplishing these goals
        - You may should clarify antecedents/symptoms in the Patient Initial Information.
    """
    if specialist_preface:
        prompt += (
            "\nThe patient is most likely experiencing diseases in your medical specialty.\n\n"
        )

    prompt += "Simply output your single question for the patient, nothing else."
    return strip_all_lines(prompt)


def get_history_taking_user_prompt(
    patient_initial_info: str = "",
    dialogue_history_text: str = "",
    conversation_goals: str = "",
):
    prompt = "Continue taking the patient's medical history by asking relevant questions, or respond 'None' if you want to end the conversation.\n\n"
    if patient_initial_info:
        prompt += f"""\
            Initial Patient Information:
            {patient_initial_info}
        """
    if conversation_goals:
        prompt += f"""
            Suggested Conversation Goals:
            {conversation_goals}\n
        """
    if dialogue_history_text:
        prompt += f"""\
            Dialogue history:
            {dialogue_history_text}\n
        """
    prompt += "Doctor: "
    return strip_all_lines(prompt)
