"""
Enhanced Professor Rating Agent with improved intent filtering.
This version has better out-of-scope detection and allows some conversational queries.
"""
from __future__ import annotations

import re
import logging
from functools import cached_property
from typing import Generator, Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import RAG
from tools.ratemyprofessor import run_tools as rmp_run

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """Configuration settings for the ProfessorRaterAgent."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    pinecone_api_key: str = Field(..., description="Pinecone API key")
    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_index_name: str = Field(
        default="professors-index",
        description="Pinecone index name",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model",
    )

    # Enhanced domain gate knobs
    enable_llm_intent_check: bool = Field(
        default=True,  # Enable LLM check for better accuracy
        description="If True, run a tiny LLM check when keyword intent is ambiguous."
    )
    allow_conversational: bool = Field(
        default=True,
        description="If True, allow basic conversational queries like 'who created you'"
    )
    out_of_scope_hint: str = Field(
        default=(
            "I'm focused on professor/course questions. "
            "Ask me about a professor's ratings, classes, or universities."
        ),
        description="Short hint added to out-of-scope replies."
    )


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    answer: str
    found_professors: bool = True


class ProfessorRaterAgent:
    """
    Enhanced Retrieval-augmented agent with improved intent filtering.
    """

    NO_PROFESSOR_MARKER = "NO PROFESSOR"

    # Enhanced intent keywords - more comprehensive
    _PROFESSOR_KEYWORDS = [
        r"\bprof(essor|s)?\b", r"\binstructor(s)?\b", r"\bteacher(s)?\b",
        r"\bfaculty\b", r"\bstaff\b"
    ]
    
    _COURSE_KEYWORDS = [
        r"\bcourse(s)?\b", r"\bclass(es)?\b", r"\bsyllabus\b",
        r"\bmidterm(s)?\b", r"\bfinal(s)?\b", r"\blecture(s)?\b",
        r"\bhomework\b", r"\bassignment(s)?\b", r"\bexam(s)?\b",
        r"\bgrade(s)?\b", r"\bgrading\b", r"\bTA\b"
    ]
    
    _EDUCATION_KEYWORDS = [
        r"\buniversity\b", r"\bcollege\b", r"\bdepartment\b",
        r"\bschool\b", r"\bcampus\b", r"\bmajor\b", r"\bdegree\b",
        r"\boffice hours?\b", r"\bprereq(uisite)?(s)?\b"
    ]
    
    _RATING_KEYWORDS = [
        r"\brating(s)?\b", r"\breview(s)?\b", r"\bevaluation(s)?\b",
        r"\bfeedback\b", r"\brate\b", r"\brmp\b", r"\bratemyprofessor\b"
    ]

    _ALL_INTENT_KEYWORDS = _PROFESSOR_KEYWORDS + _COURSE_KEYWORDS + _EDUCATION_KEYWORDS + _RATING_KEYWORDS
    _KEYWORD_REGEX = re.compile("|".join(_ALL_INTENT_KEYWORDS), re.IGNORECASE)

    # Conversational patterns that should be allowed
    _CONVERSATIONAL_PATTERNS = [
        r"\bwho (are you|created you|made you|built you)\b",
        r"\bwhat (are you|do you do|is your (name|purpose))\b",
        r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bthanks?\b", r"\bthank you\b",
        r"\bbye\b", r"\bgoodbye\b", r"\bhow are you\b", r"\bhelp\b"
    ]
    _CONVERSATIONAL_REGEX = re.compile("|".join(_CONVERSATIONAL_PATTERNS), re.IGNORECASE)

    # Explicit out-of-scope patterns (things we definitely don't want to answer)
    _OUT_OF_SCOPE_PATTERNS = [
        r"\b(car|cars|automobile|vehicle)\b",
        r"\b(movie|movies|film|tv|television)\b",
        r"\b(food|recipe|cooking|restaurant)\b",
        r"\b(weather|climate)\b",
        r"\b(stock|investment|finance|money)\b",
        r"\b(medical|health|doctor|medicine)\b",
        r"\b(legal|law|lawyer|court)\b",
        r"\b(political|politics|election|vote)\b",
        r"\b(shopping|buy|purchase|product)\b"
    ]
    _OUT_OF_SCOPE_REGEX = re.compile("|".join(_OUT_OF_SCOPE_PATTERNS), re.IGNORECASE)

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        if not self._config.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required but not set"
            )

    @cached_property
    def _pinecone_client(self) -> Pinecone:
        return Pinecone(api_key=self._config.pinecone_api_key)

    @cached_property
    def _embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=self._config.embedding_model)

    @cached_property
    def _rag(self) -> RAG:
        return RAG(
            pinecone_client=self._pinecone_client,
            pinecone_index_name=self._config.pinecone_index_name,
            embedding=self._embeddings,
        )

    @cached_property
    def _llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self._config.llm_model, temperature=0)

    @cached_property
    def _contextualize_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question which can be "
            "understood without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    @cached_property
    def _qa_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "You are the Professor Finder Bot, created by Vaidik. Your primary role is to "
            "help users find professors from an internal database. If you can't locate the "
            "information there, you can use specialized tools to search by professor names or "
            "their associated universities.\n\n"
            "STRICT DOMAIN: Only answer questions about professors, courses, teaching, or universities. "
            f"If the user question is outside this domain, reply only with \"{self.NO_PROFESSOR_MARKER}\".\n\n"
            "Use the provided context to identify professors that meet the user's criteria. "
            f"If you don't find any relevant professors, respond with \"{self.NO_PROFESSOR_MARKER}\".\n\n"
            "Communication Guidelines:\n"
            "- Use a friendly and approachable tone\n"
            "- Avoid overly formal or technical language\n"
            "- Be clear and concise in your responses\n\n"
            f"Important: If no professors are found or not enough information is found, "
            f"always end your response with \"{self.NO_PROFESSOR_MARKER}\".\n\n"
            "Context: {context}"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    @cached_property
    def _retrieval_chain(self):
        """Construct a history-aware retrieval chain (cached)."""
        history_aware_retriever = create_history_aware_retriever(
            self._llm, self._rag.get_retriever(), self._contextualize_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            self._llm, self._qa_prompt
        )
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def _is_conversational_query(self, text: str) -> bool:
        """Check if this is a basic conversational query we should handle."""
        if not self._config.allow_conversational:
            return False
        return bool(self._CONVERSATIONAL_REGEX.search(text))

    def _handle_conversational_query(self, text: str) -> str:
        """Handle basic conversational queries."""
        text_lower = text.lower().strip()
        
        if any(pattern in text_lower for pattern in ['who are you', 'what are you', 'who created you', 'who made you']):
            return "I'm the Professor Finder Bot, created by Vaidik! I help students find and learn about professors, their ratings, courses, and universities. How can I help you find information about a professor or course?"
        
        if any(pattern in text_lower for pattern in ['hello', 'hi', 'hey']):
            return "Hello! I'm here to help you find information about professors and courses. You can ask me about professor ratings, search by university, or get course information. What would you like to know?"
        
        if any(pattern in text_lower for pattern in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask me anything about professors or courses."
        
        if any(pattern in text_lower for pattern in ['help']):
            return "I can help you find professors and course information! Try asking me things like:\n- 'Find professor John Smith at Harvard'\n- 'Show me professors at MIT'\n- 'What's the rating for Professor Johnson?'\n\nWhat would you like to search for?"
        
        if any(pattern in text_lower for pattern in ['bye', 'goodbye']):
            return "Goodbye! Come back anytime you need help finding professor or course information."
        
        return "I'm here to help with professor and course questions. What would you like to know about?"

    def _is_professor_intent(self, text: str) -> bool:
        """Enhanced intent detection with multiple layers."""
        if not text or not text.strip():
            logger.info("intent=false (empty)")
            return False

        # First check: explicit out-of-scope patterns
        if self._OUT_OF_SCOPE_REGEX.search(text):
            logger.info("intent=false (explicit out-of-scope match)")
            return False

        # Second check: conversational queries (allowed if enabled)
        if self._is_conversational_query(text):
            logger.info("intent=conversational")
            return True

        # Third check: professor-related keywords
        if self._KEYWORD_REGEX.search(text):
            logger.info("intent=true (keyword match)")
            return True

        # Fourth check: LLM-based intent classification (if enabled)
        if not self._config.enable_llm_intent_check:
            logger.info("intent=false (no keyword match, LLM check disabled)")
            return False

        try:
            judge_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    (
                        "You classify if a question is about professors, courses, teaching, or universities. "
                        "Also return 'yes' for basic conversational queries like greetings or 'who are you'. "
                        "Return 'no' for questions about cars, movies, food, weather, stocks, medical advice, etc. "
                        "Answer strictly with 'yes' or 'no'."
                    ),
                ),
                ("human", "{q}"),
            ])
            chain = judge_prompt | self._llm
            out = chain.invoke({"q": text})
            content = (getattr(out, "content", "") or "").strip().lower()
            res = content.startswith("y")
            logger.info("intent via LLM=%s for query: %s", res, text[:50])
            return res
        except Exception as e:
            logger.warning("LLM intent check failed: %s -> default=false", e)
            return False

    def _nice_out_of_scope_message(self) -> str:
        return (
            f"Sorry, I can only help with professor and course related questions. "
            f"{self._config.out_of_scope_hint}"
        )

    def _pinecone_first_then_rmp(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None,
    ) -> ChatResponse:
        """
        Enhanced processing with better intent filtering.
        """
        logger.info("Processing query: %r", user_input)

        # Early exit if out of scope
        if not self._is_professor_intent(user_input):
            logger.info("Query rejected: out of scope")
            return ChatResponse(
                answer=self._nice_out_of_scope_message(),
                found_professors=False,
            )

        # Handle conversational queries
        if self._is_conversational_query(user_input):
            logger.info("Handling conversational query")
            return ChatResponse(
                answer=self._handle_conversational_query(user_input),
                found_professors=True,  # Mark as successful to avoid RMP fallback
            )

        # Proceed with normal professor search
        history_list = list(chat_history or [])

        # 1) Pinecone RAG
        result = self._retrieval_chain.invoke(
            {"input": user_input, "chat_history": history_list}
        )
        answer: str = result["answer"]
        found_professors = self.NO_PROFESSOR_MARKER not in answer
        
        if found_professors:
            return ChatResponse(answer=answer, found_professors=True)

        # 2) Fallback: RateMyProfessors
        try:
            rmp_answer = rmp_run(user_input, history_list)
            if isinstance(rmp_answer, str) and rmp_answer.strip():
                # Warm up and upsert
                try:
                    _ = self._rag.lookup("warmup")
                    if hasattr(self._rag, "_vector_store") and hasattr(
                        self._rag._vector_store, "add_texts"
                    ):
                        self._rag._vector_store.add_texts(
                            texts=[rmp_answer],
                            metadatas=[{"source": "ratemyprofessors"}],
                            ids=[f"rmp-{abs(hash(user_input))}"],
                        )
                except Exception:
                    pass
                return ChatResponse(answer=rmp_answer, found_professors=True)
        except Exception as e:
            logger.warning("RMP fallback failed: %s", e)

        return ChatResponse(answer=answer, found_professors=False)

    def invoke(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> ChatResponse:
        """Process a user query and return a completed response."""
        return self._pinecone_first_then_rmp(user_input, chat_history)

    def stream(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> Generator[str, None, None]:
        """Stream the response."""
        resp = self._pinecone_first_then_rmp(user_input, chat_history)
        yield resp.answer

    def interactive_cli(self) -> None:
        """Interactive CLI for testing."""
        print("🎓 Enhanced Professor Finder Bot - Type 'exit' to quit.")
        print("-" * 50)
        chat_history: list[BaseMessage] = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    continue
                    
                resp = self.invoke(user_input, chat_history)
                print(f"🤖 Bot: {resp.answer}")
                
                chat_history.extend([
                    HumanMessage(content=user_input), 
                    SystemMessage(content=resp.answer)
                ])
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"⚠ Error: {e}")


def create_agent(config_path: str | None = None) -> ProfessorRaterAgent:
    """Factory function to create a configured ProfessorRaterAgent."""
    config = AgentConfig(_env_file=config_path) if config_path else AgentConfig()
    return ProfessorRaterAgent(config)


if __name__ == "__main__":
    agent = create_agent()
    agent.interactive_cli()