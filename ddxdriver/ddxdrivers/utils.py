import ast
from typing import Tuple, List

from ddxdriver.utils import (
    strip_all_lines,
    ddx_list_to_string,
    DialogueHistory,
    Patient,
    Agents,
    Constants,
)
from ddxdriver.models import Model

def dialogue_to_patient_profile(dialogue_history_text: str, patient: Patient, model: Model) -> str:
    """
    Generates a patient profile representing the current dialogue history between the patient and the doctor

    Parameters:
    dialogue_history_text: text of dialogue history between doctor and patient
    model: a model which is used to generate the new patient profile

    Returns:
    Newly generated patient profile, or, if dialogue history is empty, an empty string
    """
    if not dialogue_history_text:
        raise ValueError(
            "Dialogue_history is empty, but this function requires one to update the patient profile"
        )

    system_prompt = get_dialogue_to_patient_profile_system_prompt()
    user_prompt = get_dialogue_to_patient_profile_user_prompt(
        dialogue_history_text=dialogue_history_text,
        patient=patient,
    )
    new_patient_profile = model(user_prompt=user_prompt, system_prompt=system_prompt)
    return new_patient_profile

def get_dialogue_to_patient_profile_system_prompt():
    return strip_all_lines(
        f"""\
        Your task is to transform the medical dialogue between a patient and a doctor into a comprehensive patient profile.

        Inputs:
        - Initial Patient Information (optional): The initial medical information the patient self-reported to the doctor before the dialogue.
        - Dialogue History: The medical dialogue history between you and the patient.
        - Current Patient Profile (optional): The existing patient profile for the patient, if provided.
        
        Instructions:
        1. Extract Patient Information: 
        - Extract the patient's antecedents (past medical history or conditions of the patient) and symptoms that the patient reports from the initial patient information and dialogue history.
        - Only add information present in the current profile or dialogue, do not fabricate information
        2. Create Updated Patient Profile:
        - If a Current Patient Profile is provided, integrate the new information from the Dialogue History into this profile. Modify the profile to reflect new information, but avoid removing any details unless the patient provides clear contradictory information.
        - Ensure the profile accurately reflects the patient's current state based on the latest dialogue.
        3. Format:
        - Simply format the patient profile as a dashed list, with each piece of information separated by newlines.
        - Do not include any headers or explanations, just the dashed list of antecedents/symptoms

        Here is an example format (replace the placeholders inside the brackets, and do not include the brackets themselves):

        Patient Profile:
        - [INFO_1]
        - [INFO_2]
        - ...
        
        4. Output:
        - Respond with only the updated patient profile. Do not include any additional commentary or information beyond the profile itself.
    """
    )

def get_dialogue_to_patient_profile_user_prompt(
    dialogue_history_text: str,
    patient: Patient | None = None,
):
    """
    Returns an updated patient profile based on
    """
    prompt = ""
    if patient and patient.patient_initial_info:
        prompt += f"""\
            Initial Patient Information:
            {patient.patient_initial_info}
        """
    prompt += f"""\
        Dialogue History:
        {dialogue_history_text}
    """
    if patient and patient.patient_profile:
        prompt += f"""\
            Current Patient Profile:
            {patient.patient_profile}\n
        """

    prompt += "Patient Profile:\n"
    return strip_all_lines(prompt)


def get_agent_description(
    agent_type: str, agent_prompt_length: int = Constants.AGENT_PROMPT_LENGTH.value
):
    """
    Returns function of agent specified and the prompt for the ddxdriver to pass to the agent
    Params:
        agent_type: an agent in ["diagnosis", "history_taking", "rag"] (agents Enum in ddxdriver)
        agent_prompt_length: how many topics to prompt the agent with (expected to be > 0 here)
    Output:
        agent_function (str), agent_prompt (str)
    """
    valid_agents = [agent.value for agent in Agents]
    if agent_type not in valid_agents:
        raise ValueError(
            f"Requesting description of an incorrect agent, should be one of: {valid_agents}"
        )
    # If calling this function, should prompt an agent with some positive number of topics
    if not agent_prompt_length or not isinstance(agent_prompt_length, int):
        agent_prompt_length = Constants.AGENT_PROMPT_LENGTH.value

    agent_function, agent_prompt = "", ""
    if agent_type == Agents.HISTORY_TAKING.value:
        agent_function = "Asks questions to the patient to learn more of their medical history (antecedents, symptoms)."
        agent_prompt = f"""\
            Provide at maximum {agent_prompt_length} specific conversation goals which you wish the history taker to cover.
            Design conversation goals based on the patient profile; do not repeat information already present. 
            Only include the most important topics/questions in your conversation goals.
            You may include less than the maximum of {agent_prompt_length} conversation goals.

            Design your conversation goals based on these instructions for history taking: 
            - At the beginning of the dialogue (when the patient profile is limited), you should start off asking the patient questions which determine what area of diseases they likely have. You may ask for what kind of symptoms or antecedents/diseases they mainly have. If you have a medical specialty, focus your questions in that domain.
            - As the dialogue progresses (as the patient profile grows), you should narrow your questions to specific types of antecedents/symptoms which you believe are relevant to the diseases the patient likely has
            - Ask specific questions about certain antecedents/symptoms or characteristics (age, sex, weight, etc.).
            - You may clarify chief complaints/main symptoms.
            - If Conversation Goals are specified, focus your questions on accomplishing these goals
            - You may should clarify antecedents/symptoms in the Patient Initial Information.
        """
    if agent_type == Agents.RAG.value:
        # Since rag requires at least one topic to search for, default to Constants.AGENT_PROMPT_LENGTH.value
        agent_function = "Retrieves relevant disease information (symptoms, antecedents) to help create/edit the differential diagnosis of the patient."
        agent_prompt = (
            f"Provide a detailed yet concise input search about at maximum {agent_prompt_length} diseases the patient may be suffering from.\n"
            "Search about the diseases' antecedents, symptoms, or other information which can assist in creating an differential diagnosis based on the current patient information.\n"
            "Search about the most likely diseases the patient may be suffering from (high ranks in the differential diagnosis), if not yet searched for in the previous RAG content.\n"
            "Do not search about diagnostic tests, treatments, or other external steps to take. The information should be current. \n"
            "You should also provide a free text instruction of how you want the agent to respond.\n"
            "Limit your response to length of a short paragraph or two, which may include a short list.\n"
        )
    if agent_type == Agents.DIAGNOSIS.value:
        agent_function = "Returns a ranked differential diagnosis of the patient. Either creates a new one or edits a previous one."
        agent_prompt = strip_all_lines(
            f"""\
            Provide at maximum {agent_prompt_length} evidence-based instructions to edit and guide the ranked differential diagnosis list (ddx).
            All your instructions must provide clear positive evidence of a relation between the disease and the patient (i.e. antecedents/symptoms that the patient shares with the disease)
            Ensure that you do not provide instructions based on this.
            a) Do not include instructions based on a lack of evidence. Only use the available positive evidence provided.
            b) Do not include instructions based on simple presence of the disease in the Available Information, such as in the RAG content. There must be a clear, evidence-based list.
            When providing evidence, incorporate information present in Available Information as well as your own logic.
            Select only the most important instructions which will optimize the ranked differential diagnosis list (ddx).
            Focus especially on optimizing the higher ranks of the ddx.
            You may include less than the maximum of {agent_prompt_length} instructions.

            You may choose any of the following instruction options.
            If applicable and optimal, try to provide a variety of the instruction options.
            Instruction options:
            1. Add
            - Add a likely disease to the ddx
            - You can call Add if the disease is not present in the ddx
            - Used if the disease warrants being considered as a likely candidate
            - If already present in the ddx, can call Prioritize to increase the ranking
            2. Remove
            - Remove an unlikely disease from the ddx
            - You can call Remove whether or not the disease is present in the ddx
            - Only used if the disease appears to not be a likely at all given the current information
            - Can call Deprioritize if it doesn't warrant removal, but appears less likely
            3. Prioritize
            - Rank this likely disease higher in the ddx
            - You can call Prioritize whether or not the disease is present in the ddx
            4. Deprioritize
            - Rank this disease lower in the ddx
            - You can call Deprioritize whether or not the disease is present in the ddx
            5. Maintain
            - Keeping a disease in its current position in the ddx
            - You can call Deprioritize only if the disease is present in the ddx
            - Mostly used for the top most likely diseases in the ddx, indicating confidence in the current ranking given the surrounding diseases
            
            You do not need to provide each of these categories. Instead, just provide a numbered list of edits.
            Do not output a ranked differential diagnosis of your own in these instructions, not even a revised one.
            
            I will provide an example below of how to format your instructions as a numbered list of some instruction choices above with evidence after each choice. 
            You may choose to generate up to {agent_prompt_length} instructions.
            Replace:
            - [INSTRUCTION_OPTION] with one of the instruction options above.
            - [INSTRUCTION_CONTEXT] with a disease name/type + optional context about reranking/other specifics in the ranked differential diagnosis
            - [EVIDENCE] with the evidence for your suggestion
            
            Format example (replace the placeholders inside the brackets, and do not include the brackets themselves):
            1. **[INSTRUCTION_OPTION] [INSTRUCTION_CONTEXT]**: [EVIDENCE]
            2. **[INSTRUCTION_OPTION] [INSTRUCTION_CONTEXT]**: [EVIDENCE]
            ...
            ({agent_prompt_length} instructions)

            Notes on providing instructions: 
            Do: 
            1) Provide instructions for editing the ranked differential diagnosis
             - If previous ranked differential diagnoses are provided, you should emphasize edit suggestions for it. 
             - If previous ranked differential diagnoses are not provided, you may treat this as the first ranked differential diagnosis
            2) Provide specific evidence for your above suggestions, drawn from the RAG content and dialogue history
            - This evidence may include important antecedents/symptoms of the patient, or relevant disease information from the RAG content
            - This information should be specific and not require additional diagnostic testing
            3) Only suggest the diseases specified
            - If a diagnosis options list is specified, you must use the exact terminology from the list when referring to the diseases
            - If no diseases are specified, you may choose from all diseases (probably mostly within your medical specialty)
            4) Only make suggestions based on clear evidence

            Do not:
            1) Output a ranked differential diagnosis, just the edit instructions
            2) Provide instructions without any evidence
            3) Repeat too much information from the patient initial information and patient profile. The medical agent already knows this information.
            4) Suggest diagnositc tests, only information that can be used currently for diagnosis

            Instructions for disease general categories/subcategories:
            - Sometimes both general disease categories and disease subcategories exist, or are enumerated in the diagnosis options list.
            - Only diagnose a general or subcategory if they exist. 
            - Do not repeat the same general category/subcategory.
            - You may diagnose both general categories and subcategories in the same differential diagnosis.
            - Prioritize correctly diagnosing a more specific disease subcategory (if applicable) rather than a general category.
            - For the most likely diseases, add its applicable subcategories & general category close together in the ranking.

            Here are some examples correctly and incorrectly following these instructions. Do not overfit to the content in these examples.

            Correct example: 
            Diagnosis options include: Granuloma annulare and Localized granuloma annulare. 
            Patient context: the patient seems to most likely exhibit Localized granuloma annulare, as well as general granuloma annulare.
            Ranked Differential Diagnosis:
            1. Localized granuloma annulare
            2. Granuloma annulare
            ...
            Reasoning: The diagnosis options include a general category (Granuloma annulare) and subcategory (Localized granuloma annulare). Since the patient exhibits the subcategory, the differential diagnosis correctly includes both and priortizes the subcategory. 

            Correct example: 
            Diagnosis options include: Tyrosinemia type 1, Tyrosinemia type 2, Tyrosinemia type III
            Patient context: the patient seems to exhibit either Tyrosinemia type 1 or 2, not III. Type 1 appears more likely.
            Ranked Differential Diagnosis:
            1. Tyrosinemia type 1
            2. Tyrosinemia type 2
            3. Phenylketonuria
            ...
            Reasoning: The diagnosis options include three types of Tyrosinemia, but not general Tyrosinemia. Since the patient appears most likely to exhibit type 1 then 2, but not III, only type 1 and 2 are included. The general category of Tyrosinemia is not included because it is not a diagnosis option. 

            Incorrect Example:
            Diagnosis options include: Bronchitis, Bronchiolitis, Bronchiectasis, Bronchospasm / acute asthma exacerbation
            Patient context: the patient seems to exhibit trouble with their airways, pointing to some disease of the Bronchus.
            Ranked Differential Diagnosis:
            1. Bronchitis
            2. Bronchiolitis
            3. Bronchiectasis
            4. Bronchospasm / acute asthma exacerbation
            5. Bronchiolitis
            Reasoning: The ranked differential diagnosis repeats Bronchiolitis, which is not allowed. However, it does correctly include related subcategories of diseases from the diagnosis options, grouping them together near the top of the list.\
            """
        )
    return agent_function, agent_prompt


def get_fixed_choice_system_prompt(
    specialist_preface: str = "",
) -> str:
    specialist_preface_prompt = (
        specialist_preface if specialist_preface else "You are a medical doctor."
    )
    return strip_all_lines(
        f"""\
            {specialist_preface_prompt}
            Your job is to facilitate the process of differential diagnosis of a patient by concisely prompting medical agents.
            
            You will be provided with: 
            1) Agent Descriptions. This includes:
            a) Agent Function: A description of the function of medical agent.
            b) Agent Prompt: A description of how to prompt the agent
            2) Available Information: The available information you can extract from to prompt the agent. Do not invent new information. This may include: 
                a) Patient Initial Information
                b) Patient Profile
                c) Dialogue History
                d) Previous RAG content
                - External information found about diseases the patient may be suffering from
                e) Previous Ranked Differential Diagnoses
                f) Diagnosis Options 
                - These are the only diseases the patient may be suffering from.
                - You must use the exact terminology in this list when referring to the diseases
           
            Follow these steps to create a prompt for the medical agent:
            1. Analyze the description of the medical agent and its input prompt. Note whether its input prompt is optional or mandatory.
            2. Review the current information you were provided. Determine how this information can help the agent.
            - You should only prompt based on this current information.
            3. Follow the agent's input prompt description and design a prompt for this agent.
            4. Respond with your agent prompt, nothing else.
       """
    )


def get_fixed_choice_user_prompt(
    agent_type: str,
    patient: Patient | None = None,
    dialogue_history_text: str = "",
    rag_content: str = "",
    previous_pred_ddxs: List[List[str]] = [],
    diagnosis_options: List[str] = [],
    agent_instruction_length: int = Constants.AGENT_PROMPT_LENGTH.value,
    rag_search_length: int = Constants.AGENT_PROMPT_LENGTH.value,
) -> str:
    valid_agents = [agent.value for agent in Agents]
    if not agent_type or agent_type not in valid_agents:
        raise ValueError(
            f"Error calling get_fixed_chocie_user_prompt: agent_type is None or not in {valid_agents}"
        )
    if (
        not patient
        and not dialogue_history_text
        and not rag_content
        and not previous_pred_ddxs
        and not diagnosis_options
    ):
        raise ValueError(
            "Trying to prompt agent but without any context, all context parameters are empty"
        )

    agent_prompt_length = (
        agent_instruction_length if agent_type != Agents.RAG.value else rag_search_length
    )
    agent_function, agent_prompt = get_agent_description(
        agent_type=agent_type, agent_prompt_length=agent_prompt_length
    )
    prompt = f"""\
        Agent Descriptions:

        Agent function: {agent_function}
        Agent prompt:
        {agent_prompt}

        Available Information:\n
    """
    if patient and patient.patient_initial_info:
        prompt += f"Patient Initial Information:\n{patient.patient_initial_info}\n\n"
    if patient and patient.patient_profile:
        prompt += f"Patient Profile:\n{patient.patient_profile}\n\n"
    if dialogue_history_text:
        prompt += f"Dialogue History:\n{dialogue_history_text}\n\n"
    if rag_content:
        prompt += f"Previous RAG Content:\n{rag_content}\n\n"
    if previous_pred_ddxs:
        previous_pred_ddx_str = "\n".join(
            [ddx_list_to_string(pred_ddx) for pred_ddx in previous_pred_ddxs]
        )
        prompt += f"Previous Ranked Differential Diagnoses:\n{previous_pred_ddx_str}\n\n"
    if diagnosis_options:
        diagnosis_options_str = ", ".join(diagnosis_options)
        prompt += "You may only choose diagnoses from the following list of diagnosis options (which may include synonyms/related diseases), using the exact disease terminology below:\n"
        prompt += f"Diagnosis Options:\n{diagnosis_options_str}\n\n"
    prompt += "Agent prompt:\n"
    return strip_all_lines(prompt)


def get_open_choice_system_prompt(
    specialist_preface: str = "",
    max_turns: str = "",
) -> str:
    specialist_preface_prompt = (
        specialist_preface if specialist_preface else "You are a medical doctor."
    )
    return strip_all_lines(
        f"""\
            {specialist_preface_prompt}
            Your job is to facilitate the process of differential diagnosis of a patient by:
            1) Choosing a medical agent among the available agents
            2) Concisely prompting the medical agent

            This differential diagnosis will be done iteratively, where you choose different medical agents over time.
            You will have a total of {max_turns} turns to select available medical agents. 
            Select each agent at least once during these turns (if provided enough turns).
            Select and prompt the medical agent for the most useful information (antecedents, symptoms, diagnoses, etc.) at this time step. This should change over different time steps.
            
            You will be provided with:
            1) Available agents: The medical agents you can choose from
            2) Agent Descriptions. For each available agent, this includes:
                a) Agent Name: The name of the agent. You will use this name to choose the agent.
                b) Agent Function: A description of the function of medical agent.
                c) Agent Prompt: A description of how to prompt the agent
            3) Turn Number: the current turn you are on to chose the agents
            4) Agent Choice History: a numbered list of your past agent choices (larger numbers more recent choices)
            5) Available Information: The available information you can extract from to prompt the agent. Do not invent new information. This may include:
                a) Patient Initial Information
                b) Patient Profile
                c) Dialogue History
                d) Previous RAG content
                - External information found about diseases the patient may be suffering from
                e) Previous Ranked Differential Diagnoses
                f) Diagnosis Options
                - These are the only diseases the patient may be suffering from.
                - You must use the exact terminology in this list when referring to the diseases

            Instructions on choosing among multiple available agents:
            - Choose whichever agent you think will provide the most useful information at this step.
            - Choose each agent at least once, if provided enough turns.
            - Remember that you only have {max_turns} to select agents. 
            - You may chose any available agent multiple times, but do not repeat a single agent choice repeatedly.
            - You do not need to choose agents in a specific order.
            - If {Agents.HISTORY_TAKING.value} is available:
                - You must choose this agent one or more times in order to learn more about the patient's antecedents/symptoms
                - You must choose {Agents.HISTORY_TAKING.value} at least once before choosing {Agents.DIAGNOSIS.value}. 
                - If Patient Initial Information and Patient Profile are not provided, then you must choose {Agents.HISTORY_TAKING.value} first
                - You may still call other agents besides {Agents.HISTORY_TAKING.value}

            Follow these steps to choose a medical agent and create a prompt for it:
            1. Analyze each of the available medical agents. Look at their function and input prompt
            2. Review the current information you were provided. Determine how this information warrants choosing a specific available agent.
            3. Choose one of the available agents, based on which can provide the most useful information at this step
            4. Follow the agent's input prompt description and design a prompt for this agent
            5. Respond with your agent choice and prompt in the following format, with no additional text
            - Replace [AGENT_CHOICE] with one of the agent names from the Available Agents: rag, history_taking, diagnosis
            - Replace [AGENT_PROMPT] with a prompt following the prompt instructions for that agent

            Response Format (replace the placeholders [AGENT_CHOICE] and [AGENT_PROMPT] inside the brackets, and do not include the brackets themselves):
            [AGENT_CHOICE]
            {'-'*Constants.DASH_NUMBER.value}
            [AGENT_PROMPT]

            Directly provide your response in the format specified, without additional text or brackets.
       """
    )


def get_open_choice_user_prompt(
    turn_number: int,
    available_agents: List[str],
    patient: Patient | None = None,
    dialogue_history_text: str = "",
    rag_content: str = "",
    previous_pred_ddxs: List[List[str]] = [],
    diagnosis_options: List[str] = [],
    agent_choice_history: List[str] = [],
    agent_instruction_length: int = Constants.AGENT_PROMPT_LENGTH.value,
    rag_search_length: int = Constants.AGENT_PROMPT_LENGTH.value,
) -> str:
    valid_agents = [agent.value for agent in Agents]
    if (
        not patient
        and not dialogue_history_text
        and not rag_content
        and not previous_pred_ddxs
        and not diagnosis_options
    ):
        raise ValueError(
            "Trying to prompt agent but without any context, all context parameters are empty"
        )

    prompt = f"\nTurn Number: {turn_number} \n"
    prompt += "\nAgent Choice History:\n"
    if agent_choice_history:
        for i, agent in enumerate(agent_choice_history, start=1):
            prompt += f"{i}. {agent}\n"
    else:
        prompt += "None\n"

    available_agents_str = "\n".join(available_agents)
    prompt += f"Available Agents:\n{available_agents_str}\n\nAgent Descriptions:\n\n"

    for i, agent_type in enumerate(available_agents, start=1):
        if agent_type == Agents.RAG.value:
            agent_prompt_length = rag_search_length
        else:
            agent_prompt_length = agent_instruction_length

        agent_function, agent_prompt = get_agent_description(
            agent_type=agent_type, agent_prompt_length=agent_prompt_length
        )
        prompt += f"""\
            #{i}
            Agent name: {agent_type}
            Agent function: {agent_function}
            Agent prompt:
            {agent_prompt}\n
        """

    prompt += "\nAvailable Information:\n"
    if patient and patient.patient_initial_info:
        prompt += f"Patient Initial Information:\n{patient.patient_initial_info}\n\n"
    if patient and patient.patient_profile:
        prompt += f"Patient Profile:\n{patient.patient_profile}\n\n"
    if dialogue_history_text:
        prompt += f"Dialogue History:\n{dialogue_history_text}\n\n"
    if rag_content:
        prompt += f"Previous RAG Content:\n{rag_content}\n\n"
    if previous_pred_ddxs:
        previous_pred_ddx_str = "\n".join(
            [ddx_list_to_string(pred_ddx) for pred_ddx in previous_pred_ddxs]
        )
        prompt += f"Previous Ranked Differential Diagnoses:\n{previous_pred_ddx_str}\n\n"
    if diagnosis_options:
        diagnosis_options_str = ", ".join(diagnosis_options)
        prompt += "You may only choose diagnoses from the following list of diagnosis options (which may include synonyms/related diseases), using the exact disease terminology below:\n"
        prompt += f"Diagnosis Options:\n{diagnosis_options_str}\n\n"
    prompt += "Now, output your agent choice and agent prompt in the format specified.\n"
    return strip_all_lines(prompt)


def parse_open_choice_generation(text: str) -> Tuple[str, str]:
    """
    Input: text hopefully in this format:
    [AGENT_CHOICE]
    {'-'*30}
    [AGENT_PROMPT]

    Output: [AGENT_CHOICE], [AGENT_PROMPT]

    Raises ValueError if format incorrect
    """
    error_string = strip_all_lines(
        f"""\
        Error: Expects text in this format (replace the placeholders inside the brackets, and do not include the brackets themselves):
        [AGENT_CHOICE]
        {'-'*30}
        [AGENT_PROMPT] 
    """
    )
    parts = text.split("-" * 30)

    if len(parts) < 2:
        raise ValueError(error_string)

    agent_choice_section = parts[0]
    agent_prompt_section = parts[1]

    agent_choice = strip_all_lines(agent_choice_section)
    agent_prompt = strip_all_lines(agent_prompt_section)
    return agent_choice, agent_prompt
