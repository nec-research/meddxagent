from typing import List
from re import search, DOTALL
from ast import literal_eval

from ddxdriver.utils import strip_all_lines

MAX_KEYWORD_SEARCHES = 5

def extract_and_eval_list(string: str):
    """
    Tries to retrieve a list from a string.
    First searches for the lilst with re, then tries with ast.literal_eval, then tries to manually parse
    Can either return the list of strings, or raise an exception
    """
    # Use regex to find the list in the string with DOTALL flag
    match = search(r"\[.*?\]", string, DOTALL)

    if match:
        # Extract the matched portion (the list)
        list_str = match.group(0)
        try:
            # Try to parse the list using literal_eval
            return_list = literal_eval(list_str)
            return return_list
        except (SyntaxError, ValueError):
            # If literal_eval fails, attempt to manually parse the list
            # Remove the brackets and split by commas
            list_str = list_str.strip("[]")
            return_list = [item.strip() for item in list_str.split(",")]
            if not return_list:
                raise ValueError("List is empty or could not be parsed correctly")
            return return_list
    else:
        raise ValueError("No list found in the provided string")


def get_create_keywords_user_prompt(
    input_search: str, max_keyword_searches: int = MAX_KEYWORD_SEARCHES
):
    return strip_all_lines(
        f"""\
            Your job is assist in the creation of a differential diagnosis for a patient by searching for relevant information online.
            Given an input search from a user, break it up into a list of simplified keyword searches to find relevant medical information online. 
            Follow these steps:
            1) Determine the most important medical concepts in the input search, at most {max_keyword_searches}
            - Likely, the most important topics are at the beginning of the search and refer to specific diseases
            2) Simplify those concepts into basic keyword searches, using the keywords present in the search and possibly other relevant synonyms
            - These individual keyword searches should be very simple and short, with minimal keywords
            - They should be simple enough to yield relevant online results
            - They should capture the unique medical terminology of the search, including descriptions like "symptoms" or "antecedents"
            - They should not include words like "differential diagnosis", since this is the main task you are assisting
            3) Return these keyword searches as a list of strings (max size of {max_keyword_searches}), which should be parsable as a list
            - Only return this list, nothing else
            - If there are no relevant keywords, simply return an empty list: []

        Here is an example of the format you should follow (replace the placeholders inside the arrow brackets, and do not include the arrow brackets themselves). 
        Do not overfit to the length of the keyword list in this example.
        Format example:
        Input search:
        <INPUT_SEARCH>
        Keyword searches list:
        [<KEYWORD_SEARCH_1>, <KEYWORD_SEARCH_2>]

        Now it is your turn to break up the user's input search into a list of keyword searches.
        Only return the keyword searches list, nothing else.
        
        Input search:
        {input_search}
        Keyword searches list:
    """
    )


def get_modify_keywords_user_prompt(input_search: str, keyword_searches: List[str]):
    return strip_all_lines(
        f"""\
        You will be given an input search about medical information from a doctor and a list of keyword searches from this input search.
        These keyword searches did not return any content because either they were too complicated, had typos, or lacked clear synonyms.
        Modify each keyword search to be more likely to yield relevant medical results.
        You should consider the input search as a ground truth search, but you may add synonyms, simplify, reword, remove words, etc.
        Do not add or remove entries to the list, just modify them.
        Thus, return a list of the same size as the keyword searches list provided, just with each one modified as specified.
        
        Here is an example of the format you should follow (replace the placeholders inside the arrow brackets, and do not include the arrow brackets themselves):
        However, not overfit to the length of the keyword list in this example.
        Input search:
        <INPUT_SEARCH>
        Keyword searches list:
        [<KEYWORD_SEARCH_1>, <KEYWORD_SEARCH_2>]
        Modified keyword searches list:
        [<MODIFIED_KEYWORD_SEARCH_1>, <MODIFIED_KEYWORD_SEARCH_2>]

        Now it is your turn to break up the user's input search into a list of keyword searches.
        Only return the modified keyword searches list, nothing else.

        Input search:
        {input_search}
        Keyword searches list:
        {keyword_searches}
        Modified keyword searches list:
    """
    )


def get_rag_synthesis_system_prompt():
    return strip_all_lines(
        f"""\
        You are a helpful research assistant to a doctor creating a differential diagnosis of a patient.
        Concisely answer the doctor's input search by analyzing and summarizing the relevant medical content in the search results.
        Only provide information about topics present in the search results, even if there are additional topics in the input search.

        Inputs:
        1. Doctor's Input Search: the search the doctor requested
        - This search may contain multiple topics
        2. Search results: the search results fetched
        - You may only answer based on topics present in these search results
        3. Diagnosis Options (optional): the possible diseases the patient may be suffering from
        - If provided, use this exact terminology to refer to the diseases


        Response Instructions:
        - Ground your answer in the new information from the search results. 
        - Only provide answers for topics specified both in the input search and search results.
        - If asked about topics/diseases not mentioned in the search results, please indicate that this information was not found in the search results.
        - Only include information which can help the doctor diagnose the patient at the current moment
        - Do not include diagnostic tests or treatment options
        - Your answer should be at most a few short paragraphs
        - If there is some error (such as no input search results), simply respond with the empty string: ''
        - Do not use words such as 'the search results'.

        Please output your response to the doctor, nothing else. 
    """
    )


def get_rag_synthesis_user_prompt(
    input_search: str, search_results_text: str, diagnosis_options: List[str] = []
):
    if not input_search or not search_results_text:
        raise ValueError(
            "Trying to create rag synthesis user prompt without input search or search results"
        )

    prompt = f"Doctor's Input Search:\n{input_search}\n\n"

    if diagnosis_options:
        diagnosis_options = ", ".join(diagnosis_options)
        prompt += f"Diagnosis Options:\n{diagnosis_options}\n\n"

    prompt += f"Search Results:\n{search_results_text}"

    return strip_all_lines(prompt)
