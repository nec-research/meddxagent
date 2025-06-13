from typing import final, List, Dict
import traceback

from ddxdriver.models import init_model
from ddxdriver.utils import OutputDict, Constants

from ._searchrag_utils import (
    Corpus,
    api_search,
)
from .utils import (
    extract_and_eval_list,
    get_create_keywords_user_prompt,
    get_modify_keywords_user_prompt,
)
from ddxdriver.logger import log
from .base import RAG


class SearchRAGBase(RAG):
    def __init__(self, rag_agent_cfg):
        super().__init__(rag_agent_cfg=rag_agent_cfg)

        corpus_name = self.config.get("corpus_name", Corpus.PUBMED.value)
        self.corpus_name = (
            corpus_name if corpus_name in Corpus._value2member_map_ else Corpus.PUBMED.value
        )

        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    @final
    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> str:
        """
        Given an input search, asks gpt to return a list of keywords to search, then loops through them for results.
        Removes from list if successful. If not, then adds to another list and asks gpt to simplify
        Repeats this logic for Constants.RAG_RETRIES.value times
        Given the retrieved documents, somehow returns a summary of them
        Returns dictionary in this form:
        {OutputDict.RAG_CONTENT : "<rag content>"}
        """
        retry_counter = 0
        user_prompt = get_create_keywords_user_prompt(
            input_search=input_search, max_keyword_searches=self.max_keyword_searches
        )
        # log.info(user_prompt)
        # exit()
        # log.info(user_prompt)
        keyword_searches = []
        message_history = []
        while retry_counter <= Constants.RAG_RETRIES.value:
            try:
                keyword_searches_str = self.model(
                    user_prompt=user_prompt, message_history=message_history
                )
                # log.info(keyword_searches_str)
                # exit()
                keyword_searches = extract_and_eval_list(string=keyword_searches_str)
                keyword_searches = keyword_searches[: self.max_keyword_searches]
            except Exception as e:
                log.info(
                    f"Caught exception trying to generate/parse keyword searches list, trying again: {e}\n"
                )
                keyword_searches = []
                if retry_counter <= Constants.RAG_RETRIES.value:
                    log.info(f"Current keywords list: {keyword_searches_str}, trying again...\n")
                else:
                    log.info(f"Out of retries for keywords, returning '' for rag content...\n")
                    return ""

            if keyword_searches and all(isinstance(x, str) for x in keyword_searches):
                break

            retry_counter += 1
            message_history.extend(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": keyword_searches_str},
                ]
            )
            user_prompt = (
                "Your keyword list was not formatted correctly as a list of strings. "
                "Please edit its format so it can be parsed as such.\n"
                "Here is an example of formatting (replace the placeholders inside the arrow brackets, and do not include the arrow brackets themselves):\n"
                """["<KEYWORD_SEARCH_1>", "<KEYWORD_SEARCH_2>"]"""
            )

        if not keyword_searches or not all(isinstance(x, str) for x in keyword_searches):
            error_message = "Keyword searches list wasn't correctly generated in time\n"
            raise ValueError(error_message)

        # Searching for results
        retry_counter = 0
        search_results: List[Dict[str, str]] = []
        while keyword_searches and retry_counter < Constants.RAG_RETRIES.value:
            log.info(f"Executing keyword searches...\n{keyword_searches}\n")
            index = 0
            while index < len(keyword_searches):
                keyword_search = keyword_searches[index]
                results = []
                try:
                    results = api_search(
                        query=keyword_search, top_k=self.top_k_search, corpus_name=self.corpus_name
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    # Log exception (full traceback)
                    log.error(f"SearchRAGBase() failed due following error:\n{e}\nTraceback:\n{tb}")
                if results:
                    search_results.extend(results)
                    # Remove the item
                    keyword_searches.pop(index)
                else:
                    # Go to the next item in the list
                    index += 1

            retry_counter += 1

            # If still exist keyword_searches which were not removed from the list, try to reformat them
            if keyword_searches and retry_counter <= Constants.RAG_RETRIES.value:
                log.info(
                    f"Could not find search results for keyword searches: {keyword_searches}, will try to reformat searches...\n"
                )
                keyword_searches_str = self.model(
                    user_prompt=get_modify_keywords_user_prompt(
                        input_search=input_search, keyword_searches=keyword_searches
                    )
                )
                try:
                    keyword_searches = extract_and_eval_list(string=keyword_searches_str)
                    keyword_searches = keyword_searches[: self.max_keyword_searches]
                except Exception as e:
                    log.info(
                        f"Caught exception trying to modify keyword searches list, trying again: {e}\n"
                    )
                    keyword_searches = []

        if not search_results:
            log.info("Could not find search results, returning '' for rag...")
            return ""

        log.info("Successfully found search results\n")
        # log.info("Search results\n", search_results)
        # exit()
        rag_content = self.synthesize_results(
            input_search=input_search,
            search_results=search_results,
            diagnosis_options=diagnosis_options,
        )
        log.info("Rag content\n" + rag_content + "\n")
        return {OutputDict.RAG_CONTENT: rag_content}
