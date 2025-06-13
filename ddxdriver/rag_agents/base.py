from abc import ABC, abstractmethod
from typing import List, Dict

class RAG(ABC):
    @abstractmethod
    def __init__(self, rag_agent_cfg):
        "Initialize the rag agent"
        
        self.config = rag_agent_cfg
        self.top_k_search = self.config.get("top_k_search", 2)
        self.max_keyword_searches = self.config.get("max_keyword_searches", 5)

    @abstractmethod
    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> str:
        """
        Retrieves relevant documents given the input text, then returns content from the output (an "answer")
        Params:
            input_search (str): Full search of information the doctor wants (can be formatted in free text)

        Output:
            (str) Text answering the doctor's input search
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize_results(
        self,
        input_search: str,
        search_results: List[Dict[str, str]],
        diagnosis_options: List[str] = [],
    ):
        """
        Given new search results, synthesizes them and returns rag output as a string (using summarization, extra retrieval, etc)
        Search results will be formatted as list of dictionaries in this form:
        {
            "title": <title of search result>
            "content": <content of search result>
        }
        """
