from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model
from langchain_community.vectorstores import LanceDB

from dtos.Metadata import Metadata
from dtos.Question import Question
from dtos.OllamaResponse import OllamaResponse

import logging
import configparser


logging.basicConfig()
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)


# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")


class OllamaService:
    def __init__(self, lance_db, model, embeddings):
        # Setup LangChain
        self.llm = ChatOpenAI(
            base_url=config["config"]["ollama_uri_v1"],
            model=model,
            temperature=0,
            api_key="placeholder",
        )
        self.embeddings = OpenAIEmbeddings(
            base_url=config["config"]["ollama_uri_emb"], model=embeddings, api_key="placeholder", check_embedding_ctx_length=False
        )
        self.system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer, state that clearly.
	Format your response nicely in markdown.\n\n
        {context}
        """

        # Predefined directories
        self.lance_db = lance_db

        self.retrieval_chain = self.initialize_retrieval_chain()

    @staticmethod
    def seconds_to_hms(seconds):
        if seconds == "":
            return ""
        else:
            seconds = int(seconds)
            return (
                f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
            )

    def initialize_retrieval_chain(self):
        # Create the retriever
        vector_index = LanceDB(
            uri=self.lance_db,
            embedding=self.embeddings,
            distance="cosine",
            mode="append",
            table_name=config["vectorizer"]["table_name"],
        )

        retriever = vector_index.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6, "k": 5, "metrics": "cosine"},
        )

        # Simplify the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        retrieval_chain = create_retrieval_chain(
            retriever, question_answer_chain
        )

        return retrieval_chain

    def generate_response(self, question: Question, session_id: str) -> OllamaResponse:
        # no session history needed anymore
        with get_openai_callback() as cb:
            response = self.retrieval_chain.invoke(
                {"input": question.question}
            )

        context = []

        for element in response["context"]:
            context.append(Metadata(**element.metadata))

        formatted_response = OllamaResponse(
            context=context,
            response=response["answer"],
            tokens_out=cb.completion_tokens,
            tokens_in=cb.prompt_tokens,
            total_tokens=cb.total_tokens,
            saved_costs=get_openai_token_cost_for_model("gpt-4", cb.total_tokens),
            session_id=session_id,
        )

        print("Tokens in: " + str(formatted_response.tokens_in))
        print("Tokens out: " + str(formatted_response.tokens_out))
        print("Tokens total: " + str(formatted_response.tokens_out))
        print("Generated response: " + formatted_response.response)

        return formatted_response
