from ddxdriver.models import init_model
from ddxdriver.utils import OutputDict

from ._medrag_utils import Retriever
from .base import RAG
from .utils import get_rag_synthesis_system_prompt, get_rag_synthesis_user_prompt


class MedRAG(RAG):
    """
    Using medrag data
    Code not being used (not fully updated)
    """
    def __init__(self, rag_agent_cfg):
        self.config = rag_agent_cfg
        corpus_name = self.config.get("corpus_name", "PubMed")
        self.retriever = Retriever(corpus_name=corpus_name)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    def __call__(
        self,
        input_search: str,
        diagnosis_options: List[str] = [],
    ) -> str:
        print(input_search)
        exit()
        documents = self.retriever.get_relevant_documents(question=input_search, k=5)[0]
        documents_text = "\n\n".join(
            f"{i}. {document['title']}\n\n{document['content']}"
            for i, document in enumerate(documents)
        )
        # print(documents_text)
        # exit()
        system_prompt = get_rag_synthesis_system_prompt()
        print("System prompt:\n\n" + system_prompt + "\n\n")
        user_prompt = get_rag_synthesis_user_prompt(
            input_search=input_search,
            search_results_text=documents_text,
            diagnosis_options=diagnosis_options,
        )
        print("User prompt:\n\n" + user_prompt + "\n\n")
        output = self.model(user_prompt=user_prompt, system_prompt=system_prompt)
        print("RAG output:\n\n" + output + "\n\n")
        # exit()
        return {OutputDict.RAG_CONTENT: output}
