from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model
from langchain_community.vectorstores import LanceDB

from dtos.Metadata import Metadata
from dtos.Question import Question
from dtos.OllamaResponse import OllamaResponse

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
        # store conversation history
        self.conversation_history_store = {}

        self.retrieval_chain = self.initialize_retrieval_chain()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.conversation_history_store:
            self.conversation_history_store[session_id] = ChatMessageHistory()
        return self.conversation_history_store[session_id]

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

        history_aware_retriever = self.__initialize_contextual_aware_retriever(
            retriever
        )

        # Implementation based on:
        # https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#stateful-management-of-chat-history
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        retrieval_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return retrieval_chain

    def __initialize_contextual_aware_retriever(self, retriever):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Focus rather on the new question than the history unless there's a clear "
            "relation. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        return history_aware_retriever

    def generate_response(self, question: Question, session_id: str) -> OllamaResponse:
        # add session history to chain
        conversational_rag_chain = RunnableWithMessageHistory(
            self.retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        # In case the session_id does not exist, it is created
        with get_openai_callback() as cb:
            response = conversational_rag_chain.invoke(
                {"input": question.question},
                config={"configurable": {"session_id": session_id}},
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
