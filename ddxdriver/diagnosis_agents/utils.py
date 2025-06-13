from enum import Enum
from typing import List, Dict, Tuple, Any
import json

from ddxdriver.models.base import Model
from ddxdriver.benchmarks.base import Bench
from ddxdriver.utils import strip_all_lines, ddx_list_to_string, Constants


class FewshotType(Enum):
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class DiagnosisError(Enum):
    COT_FORMAT = "cot_format"
    DDX_LENGTH = "ddx_length"
    DDX_FORMAT = "ddx_format"


def get_ddx_system_prompt(
    diagnosis_options: List[str] | None = None,
    specialist_preface: str | None = None,
    ddx_length: int | None = None,
    use_cot: bool = False,
) -> str:
    """
    Generates a system prompt to provide the ranked differential diagnosis
    Diagnosis options should be a comma separated list of strings
    """
    ddx_length_prompt = f"top {ddx_length}" if ddx_length is not None else "top"
    specialist_preface_prompt = (
        specialist_preface if specialist_preface else "You are a medical doctor."
    )
    prompt = f"""\
        {specialist_preface_prompt}
        Given a patient's profile (a list of antecedents and symptoms), provide a ranked differential diagnosis of the {ddx_length_prompt} most likely diseases. 
        
        You may be provided a list of diagnosis options you can choose from. You must use this exact disease terminology when referring to the diseases.
        If you aren't provided the diagnosis options, consider all possible diseases. 

        Your ranked differential diagnosis should have the possible diseases ranked from most likely to least likely.
        
        You will also be provided with:
        1. Previous Ranked Differential Diagnoses: The previous ranked lists you generated, or None if this can be treated as the first differential diagnosis
        -  If provided Previous Ranked Differential Diagnoses, try to improve them for a new ranked differential diagnosis by adding/removing/reranking/maintaining/prioritizing diseases.
        -  Try to optimize the Previous Ranked Differential Diagnosis, focusing on moving most likely diseases upwards in the list. However, only make changes if you believe they will improve the accuracy of the diagnosis.
        2. Suggested Diagnosis Instructions (optional):
        -  Strongly consider these suggested instructions and their provided evidence when creating or editing the ranked differential diagnosis.
        -  However, if you disagree with an instruction or its evidence, you may choose not to follow it or only apply it limitedly.
        -  Ultimately, make your own diagnosis decisions about the ranked differential diagnosis, including its diseases and edits to previous differential diagnosis.
        -  Your edits should focus on optimizing the ranked differential diagnosis.
        3. Previous Search Content (optional):
        -  This may contain useful information found online about diseases the patient may be suffering from
        -  Consider this disease information when creating your diagnosis. However, only include these diseases in the ranked differential diagnosis if there is a clear connection between them and the patient.
        4. Patient profile: the known symptoms/antecedents of the patient
        5: Patient examples (optional): other relevant examples of patients with similar patient profiles as the one you are diagnosing.
        - These are relevant examples of patients, potentially with similar profiles and diseases as the patient you are trying to diagnose
        - Strongly consider the content in these examples when creating your ranked differential diagnosis. However, do not overfit to these examples.
        - If Ranked Differential Diagnoses are provided in the examples, consider making your Ranked Differential Diagnosis a similar length.
        
        Synthesize all this provided information and create the most optimal ranked differential diagnosis.\n

        Instructions for disease general categories/subcategories:
        - Sometimes both general disease categories and disease subcategories exist, or are enumerated in the diagnosis options list.
        - Only diagnose a general or subcategory if they exist. 
        - Do not repeat the same general category/subcategory.
        - If a diagnosis options list is provided, only chose diseases from this list.
        - You may diagnose both general categories and subcategories in the same differential diagnosis.
        - Prioritize correctly diagnosing a more specific disease subcategory (if applicable) rather than a general category.
        - For the most likely diseases, add its applicable subcategories & general category close together in the ranking.

        Here are some examples correctly and incorrectly following these instructions. Do not overfit to the content in these examples.

        Correct example: The diagnosis options include a general category (Granuloma annulare) and subcategory (Localized granuloma annulare). Since the patient exhibits the subcategory, the differential diagnosis correctly includes both and prioritizes the subcategory. 
        Diagnosis options include: Granuloma annulare and Localized granuloma annulare. 
        Patient context: the patient seems to most likely exhibit Localized granuloma annulare, as well as general granuloma annulare.
        Ranked Differential Diagnosis:
        1. Localized granuloma annulare
        2. Granuloma annulare
        ...

        Correct example: The diagnosis options include three types of Tyrosinemia, but not general Tyrosinemia. Since the patient appears most likely to exhibit type 1 then 2, but not III, only type 1 and 2 are included. The general category of Tyrosinemia is not included because it is not a diagnosis option. 
        Diagnosis options include: Tyrosinemia type 1, Tyrosinemia type 2, Tyrosinemia type III
        Patient context: the patient seems to exhibit either Tyrosinemia type 1 or 2, not III. Type 1 appears more likely.
        Ranked Differential Diagnosis:
        1. Tyrosinemia type 1
        2. Tyrosinemia type 2
        3. Phenylketonuria
        ...

        Incorrect Example: The ranked differential diagnosis repeats Bronchiolitis, which is not allowed. However, it does correctly include related subcategories of diseases from the diagnosis options, grouping them together near the top of the list.
        Diagnosis options include: Bronchitis, Bronchiolitis, Bronchiectasis, Bronchospasm / acute asthma exacerbation
        Patient context: the patient seems to exhibit trouble with their airways, pointing to some disease of the Bronchus.
        Ranked Differential Diagnosis:
        1. Bronchitis
        2. Bronchiolitis
        3. Bronchiectasis
        4. Bronchospasm / acute asthma exacerbation
        5. Bronchiolitis
        
        """
    if use_cot:
        prompt += "You will also provide a step-by-step rationale to help guide the ranked differential diagnosis. This is a chain of reasoning about the most likely categories of diseases given the patient's profile. Format the step-by-step rationale as a paragraph with several sentences, not as a list of the diseases you predict or a numbered list of steps.\n\n"
    if diagnosis_options:  # Checks if list not None and non-empty
        diagnosis_options = ", ".join(diagnosis_options)
        prompt += f"""\
            You may only choose diagnoses from the following list of diagnosis options (which may include synonyms/related diseases), using the exact disease terminology below:
            Diagnosis Options:
            {diagnosis_options}\n\n"""
    prompt += f"""\
        Directly provide the ranked differential diagnosis of the {ddx_length_prompt} most likely diseases for the patient in the following format (without additional text before or after), with one diagnosis per line (replace [DIAGNOSIS_X] with the actual diagnosis name, and do not include the brackets themselves): [RANK_NUMBER]. [DIAGNOSIS]. I.e.: 
        1. [DIAGNOSIS_1]
        2. [DIAGNOSIS_2]
        ...
    """
    if use_cot:
        prompt += f"""\n\n\
            Directly provide the step-by-step rationale and ranked differential diagnosis directly in the following format (replace [STEP-BY-STEP_RATIONALE] and [DDX] with the actual content, and do not include the brackets themselves):

            [STEP-BY-STEP_RATIONALE]
            {'-'*Constants.DASH_NUMBER.value}
            Ranked Differential Diagnosis:
            [DDX]
        """
    prompt += "\nDirectly provide your response in the format specified, without additional text."

    return strip_all_lines(prompt)


def get_ddx_user_prompt(
    patient_profile: str,
    shot_dicts: List[Dict[str, str]] | None = None,
    use_cot: bool = False,
    diagnosis_instructions: str = "",
    previous_pred_ddxs: List[List[str]] = [],
    previous_search_content: str = "",
) -> str:
    """
    Inputs a patient profile, optionally a list of few shot profiles
    Shots are formatted as a dict, which should include these keys:
    {
    "patient_profile" : patient profile,
    "gt_pathology": ground truth pathology,
    "gt_ddx": List[str], ground truth differential diagnosis (Optional)
    "cot_rationale": cot rationale for this example (Optional)
    }
    """
    prompt = ""
    if previous_pred_ddxs:
        previous_pred_ddx_str = (
            "\n".join(ddx_list_to_string(pred_ddx) for pred_ddx in previous_pred_ddxs) or None
        )
        prompt += f"""\
            Previous Ranked Differential Diagnoses:
            {previous_pred_ddx_str}\n\n
        """
    if diagnosis_instructions:
        prompt += f"""\
            Suggested Diagnosis Instructions:
            {diagnosis_instructions}\n\n
        """
    if previous_search_content:
        prompt += f"""\
            Previous Search Content:
            {previous_search_content}\n\n
        """
    if shot_dicts:  # If not None and not empty
        prompt += "Here are some other relevant patient examples.\n"
        for i, shot_dict in enumerate(shot_dicts, 1):
            if "patient_profile" not in shot_dict:
                raise ValueError("Fewshot entry needs to have patient_profile")

            gt_pathology_shot = (
                f"with ground truth pathology: {shot_dict['gt_pathology']}"
                if shot_dict.get("gt_pathology", None)
                else ""
            )
            prompt += f"""\
                Example {i} {gt_pathology_shot}

                Patient Profile:
                {shot_dict["patient_profile"]}\n
            """

            ddx_shot = (
                f"{ddx_list_to_string(shot_dict['gt_ddx'])}\n"
                if shot_dict.get("gt_ddx", None)
                else "[Ranked differential diagnosis with one diagnosis per line: [RANK_NUMBER]. [DIAGNOSIS].]\n"
            )

            if use_cot:
                cot_shot = (
                    shot_dict["cot_rationale"]
                    if shot_dict.get("cot_rationale", None)
                    else "[STEP-BY-STEP_RATIONALE]"
                )
                prompt += f"""\
                    Let's think step by step to derive the most likely diagnoses.
                    {cot_shot}
                    {'-'*Constants.DASH_NUMBER.value}
                    Ranked Differential Diagnosis:
                    {ddx_shot}
                """
            else:
                prompt += f"""\
                    Ranked Differential Diagnosis:
                    {ddx_shot}\n
                """

    prompt += f"""\
        Now it is your turn to provide an updated differential diagnosis for the patient.
    
        Patient Profile:
        {patient_profile}\n
        """
    if use_cot:
        prompt += "\nLet's think step by step to derive the most likely diagnoses.\n"
    else:
        prompt += "Ranked Differential Diagnosis:\n"

    return strip_all_lines(prompt)


def parse_cot_generation(text: str) -> Tuple[str, str]:
    """
    Input: text hopefully in this format (replace the placeholders inside the brackets, and do not include the brackets themselves):
    [STEP-BY-STEP_RATIONALE]
    {'-'*30}
    Ranked Differential Diagnosis:
    [DDX]

    Output: [DDX], [STEP-BY-STEP_RATIONALE]
    If no split by dashes possible or no "Ranked Differential Diagnosis:" found, 
    returns the full text as DDX and "DEFAULT" as rationale
    """
    # Try to split by dashes
    parts = text.split("-" * 30)
    
    if len(parts) < 2:
        text_to_search = text
    else:
        text_to_search = parts[1]
        rationale = strip_all_lines(parts[0])

    # Look for ranked differential diagnosis section
    start_index = text_to_search.find("Ranked Differential Diagnosis:")
    if start_index != -1:
        start_index += len("Ranked Differential Diagnosis:")
        ddx = strip_all_lines(text_to_search[start_index:])
        return ddx, rationale if len(parts) >= 2 else "DEFAULT"
    
    # If not found, return the full original text
    return strip_all_lines(text), "DEFAULT"


def get_self_generate_ddx_cot_user_prompt(
    shot_dict: Dict[str, str],
    diagnosis_options: List[str] | None = None,
    specialist_preface: str | None = None,
    ddx_length: int | None = None,
):
    """
    Generates a cot rationale to be used as a self-genereated fewshot ex
    Diagnosis options should be a comma separated list of strings
    Shots are formatted as a dict, which should include these keys:
    {
    "patient_profile" : patient profile,
    "gt_pathology": ground truth pathology,
    "gt_ddx": List[str], ground truth differential diagnosis (Optional)
    "cot_rationale": cot rationale for this example (Optional)
    }
    """

    ddx_length_prompt = f"top {ddx_length}" if ddx_length is not None else "top"
    specialist_preface_prompt = (
        specialist_preface if specialist_preface else "You are a medical doctor."
    )
    prompt = f"""\
        {specialist_preface_prompt}
        Given a patient's profile (a list of antecedents and symptoms), as well as some ground truth information, provide a step-by-step rationale which helps to derives the {ddx_length_prompt} most likely diseases.

        Instructions for the step-by-step rationale:
        - It is a chain of reasoning about the most likely categories of diseases given the patient's profile. 
        - It should be consistent with the ground truth information provided, but the rationale should act as if it was only provided the patient profile (not ground truth information).
        - It should be grounded in the patient's profile and the disease possibilities.
        - Format the step-by-step rationale as a paragraph with several sentences, not as a list of the diseases you predict or a numbered list of steps.
        - I will provide you an example of how to format the step-by-step rationale. Ensure that you follow this structure.
        - Do not overfit to content of the example's patient profile or the step-by-step rationale, since it may be a completely different specialty than yours.
        - Directly provide the step-by-step rationale without additional text before or after.\n
    """

    if diagnosis_options:  # Checks if list not None and non-empty
        diagnosis_options = ", ".join(diagnosis_options)
        prompt += f"""\
            You may only consider diagnoses from the following list, using the exact disease terminology below:
            {diagnosis_options}\n\n"""

    prompt += f"""\
        Here is an example of how to format this step-by-step rationale should look.

        Patient Profile:
        Sex: Female, Age: 9
        - I am experiencing shortness of breath or difficulty breathing in a significant way.
        - In the last month, I have been in contact with someone infected with the Ebola virus.
        - I have a fever (either felt or measured with a thermometer).
        - I have diffuse (widespread) muscle pain.
        - I am feeling nauseous or I feel like vomiting.
        - I have noticed unusual bleeding or bruising related to my consultation today.
        - I have traveled out of the country to West Africa in the last 4 weeks.

        Ground truth pathology:
        Ebola\n
        """
    if shot_dict.get("gt_ddx"):
        prompt += f"""\
            Ranked Differential Diagnosis:
            1. Pneumonia
            2. Ebola
            3. Atrial fibrillation
            4. Guillain-Barré syndrome
            5. Acute dystonic reactions
            6. Croup
            7. HIV (initial infection)
            8. PSVT
            9. Panic attack
            10. Pulmonary embolism\n
        """

    prompt += f"""\
        Step-by-step rationale:
        The patient's demographics and recent travel history to West Africa, coupled with contact with an Ebola-infected individual, suggest a high likelihood of an infectious disease. Shortness of breath, fever, diffuse muscle pain, nausea, and unusual bleeding are key symptoms indicative of systemic involvement and potential hemorrhagic conditions. The presence of unusual bleeding particularly points towards viral hemorrhagic fevers, such as Ebola. However, other infectious diseases endemic to West Africa, like malaria and dengue, should also be considered. Non-infectious causes, though less likely, must be kept in mind.
    """
    prompt += f"""\
        \nNow it is your turn to create a step-by-step rationale given a patient profile and available ground truth information.

        Patient Profile:
        {shot_dict["patient_profile"]}

        Ground truth pathology:
        {shot_dict["gt_pathology"]}\n
    """
    if shot_dict.get("gt_ddx"):
        prompt += f"""\
        Ranked Differential Diagnosis:
        {shot_dict["gt_ddx"]}\n
        """

    prompt += "Step-by-step rationale:\n"
    return strip_all_lines(prompt)
