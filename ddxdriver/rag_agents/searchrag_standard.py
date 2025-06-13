from typing import List, Dict

from .searchrag_base import SearchRAGBase
from ._searchrag_utils import format_search_result
from .utils import get_rag_synthesis_system_prompt, get_rag_synthesis_user_prompt
from ddxdriver.logger import log

class SearchRAGStandard(SearchRAGBase):

    def synthesize_results(
        self,
        input_search: str,
        search_results: List[Dict[str, str]],
        diagnosis_options: List[str] = [],
    ) -> str:
        """
        Given new search results, synthesizes them and returns rag output to ddxdriver (using summarization, extra retrieval, etc)
        """
        if not search_results:
            raise ValueError("Trying to synthesize search results, but search results is None")
        # log.info(input_search)
        # exit()
        try:
            search_results_text = "\n\n".join(
                format_search_result(result) for result in search_results
            )
            # log.info(search_results_text)
            system_prompt = get_rag_synthesis_system_prompt()
            # log.info("System prompt:\n\n" + system_prompt + "\n\n")
            user_prompt = get_rag_synthesis_user_prompt(
                input_search=input_search,
                search_results_text=search_results_text,
                diagnosis_options=diagnosis_options,
            )
            # log.info("\nUser prompt:\n\n" + user_prompt + "\n\n")
            # exit()
            output = self.model(user_prompt=user_prompt, system_prompt=system_prompt)
            # log.info("RAG output:\n\n" + output + "\n\n")
            # exit()
            return output
        except Exception as e:
            log.info(f"Caught exception: {e}. Returning empty string...")
