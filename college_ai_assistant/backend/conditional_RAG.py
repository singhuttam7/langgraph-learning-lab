import os
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from colorama import Fore, Style, init


# ============================================================
# STEP 1 - PATH CONFIGURATION
# ============================================================

# Current file:
# F:\LangGraph\college_ai_assistant\backend\conditional_RAG.py
#
# BASE_DIR:
# F:\LangGraph\college_ai_assistant

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

ENV_FILE = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# STEP 2 - HUGGING FACE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# STEP 3 - BUILD RAG RETRIEVER
# ============================================================

def build_retriever(pdf_path: str):
    """
    Load a PDF, split it into chunks and create
    a FAISS retriever.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}"
        )

    print(f"Loading PDF: {pdf_file}")

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


# ============================================================
# STEP 4 - CREATE RETRIEVERS
# ============================================================

ACADEMIC_PDF = DATA_DIR / "academics_handbook.pdf"

FEE_PDF = DATA_DIR / "fee_structure.pdf"


academic_retriever = build_retriever(
    str(ACADEMIC_PDF)
)

fee_retriever = build_retriever(
    str(FEE_PDF)
)


# ============================================================
# STEP 5 - GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4
)


# ============================================================
# STEP 6 - STATE
# ============================================================

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# STEP 7 - CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:
    """
    Look at the latest user message and decide
    which path to take.
    """

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly "
        "one category: academic, fee, or general.\n\n"

        "Use academic for questions about attendance, exams, "
        "grading, credits, promotion, course structure, "
        "summer training, or degree requirements.\n"

        "Use fee for questions about tuition, payment, refund, "
        "late charges, scholarships, or any money-related topic.\n"

        "Use general for greetings, casual talk, or anything "
        "not related to college rules or fees.\n\n"

        f"Query: {last_message}\n\n"

        "Return ONLY one word:\n"
        "academic\n"
        "fee\n"
        "general"
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {
        "query_type": category
    }


# ============================================================
# STEP 8 - ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State) -> dict:
    """
    Retrieve relevant chunks from the academic handbook.
    """

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# STEP 9 - FEE RAG NODE
# ============================================================

def fee_rag_node(state: State) -> dict:
    """
    Retrieve relevant chunks from the fee structure PDF.
    """

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# STEP 10 - GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:
    """
    General questions do not require document retrieval.
    """

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# STEP 11 - RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:
    """
    Generate the final answer.
    """

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "unknown"
    )

    context = state["retrieved_context"]


    # --------------------------------------------------------
    # GENERAL QUESTION
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college AI assistant "
            f"talking to a {programme} student.\n\n"

            "Answer the student's question using your "
            "general knowledge.\n\n"

            f"Question:\n{query}\n\n"

            "Give a clear, helpful and friendly answer.\n"

            "Do not use Markdown formatting.\n"
            "Do not use #, *, **, bullet points, tables, "
            "or backticks.\n"

            "Return plain text only using simple paragraphs."
        )


    # --------------------------------------------------------
    # RAG QUESTION
    # --------------------------------------------------------

    else:

        prompt = (
            f"You are a college AI assistant helping a "
            f"{programme} student.\n\n"

            "Use the following information from the official "
            "college documents to answer the student's question.\n\n"

            "IMPORTANT:\n"
            "Only use information from the provided context "
            "when answering college-specific questions.\n"

            "If the answer is not present in the context, "
            "clearly say that the information is not available "
            "in the provided college documents.\n\n"

            f"Student Programme:\n{programme}\n\n"

            f"Context:\n{context}\n\n"

            f"Question:\n{query}\n\n"

            "Give a clear, friendly and precise answer.\n"

            "If the context contains different values for "
            "different programmes, use the value relevant to "
            f"{programme} when possible.\n\n"

            "Do not use Markdown formatting.\n"
            "Do not use #, *, **, bullet points, tables, "
            "or backticks.\n"

            "Return plain text only using simple paragraphs."
        )


    response = llm.invoke(prompt)

    answer = response.content.strip()

    return {
        "messages": [
            ("ai", answer)
        ]
    }


# ============================================================
# STEP 12 - ROUTER
# ============================================================

def route_query(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# ============================================================
# STEP 13 - BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


# Add nodes

graph.add_node(
    "classifier",
    classifier_node
)

graph.add_node(
    "academic_rag",
    academic_rag_node
)

graph.add_node(
    "fee_rag",
    fee_rag_node
)

graph.add_node(
    "general",
    general_node
)

graph.add_node(
    "response",
    response_node
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

graph.add_edge(
    START,
    "classifier"
)


graph.add_conditional_edges(
    "classifier",
    route_query
)


graph.add_edge(
    "academic_rag",
    "response"
)

graph.add_edge(
    "fee_rag",
    "response"
)

graph.add_edge(
    "general",
    "response"
)


graph.add_edge(
    "response",
    END
)


# Compile graph

app = graph.compile()


# ============================================================
# STEP 14 - TERMINAL UI
# ============================================================

init(autoreset=True)


def print_banner():

    print(
        "\n"
        + Fore.CYAN
        + "=" * 65
    )

    print(
        Fore.CYAN
        + "           🎓 COLLEGE AI ASSISTANT"
    )

    print(
        Fore.CYAN
        + "=" * 65
    )

    print(
        Fore.WHITE
        + "      Intelligent RAG-powered Student Assistant"
    )

    print(
        Fore.CYAN
        + "-" * 65
    )


def print_programme_menu():

    print(
        Fore.YELLOW
        + "\n📚 SELECT YOUR PROGRAMME"
    )

    print(
        Fore.WHITE
        + "-" * 40
    )

    print(
        Fore.GREEN
        + "  [1] "
        + Fore.WHITE
        + "BCA"
    )

    print(
        Fore.GREEN
        + "  [2] "
        + Fore.WHITE
        + "BBA"
    )

    print(
        Fore.GREEN
        + "  [3] "
        + Fore.WHITE
        + "B.Com (H)"
    )

    print(
        Fore.WHITE
        + "-" * 40
    )


# ============================================================
# STEP 15 - TERMINAL APPLICATION
# ============================================================

if __name__ == "__main__":

    print_banner()

    print_programme_menu()

    choice = input(
        Fore.YELLOW
        + "\nEnter your choice (1/2/3): "
        + Style.RESET_ALL
    )


    programme_map = {
        "1": "BCA",
        "2": "BBA",
        "3": "B.Com (H)"
    }


    student_programme = programme_map.get(choice)


    # Validate programme

    if student_programme is None:

        print(
            Fore.RED
            + "\n❌ Invalid choice! Defaulting to BCA."
        )

        student_programme = "BCA"


    print(
        Fore.GREEN
        + f"\n✓ Programme selected: {student_programme}"
    )


    print(
        Fore.CYAN
        + "=" * 65
    )


    print(
        Fore.WHITE
        + "\n💡 You can ask questions about academics, fees, "
        "attendance, exams, grading, etc."
    )


    print(
        Fore.YELLOW
        + "   Type 'exit' or 'quit' to close the assistant."
    )


    print(
        Fore.CYAN
        + "=" * 65
    )


    # --------------------------------------------------------
    # CHAT LOOP
    # --------------------------------------------------------

    while True:

        print()

        user_query = input(
            Fore.BLUE
            + "👤 You: "
            + Style.RESET_ALL
        )


        # Exit

        if user_query.lower().strip() in [
            "exit",
            "quit"
        ]:

            print(
                Fore.CYAN
                + "\n"
                + "=" * 65
            )

            print(
                Fore.GREEN
                + "👋 Thank you for using College AI Assistant!"
            )

            print(
                Fore.CYAN
                + "=" * 65
                + "\n"
            )

            break


        # Empty input

        if not user_query.strip():

            print(
                Fore.YELLOW
                + "⚠️ Please enter a question."
            )

            continue


        # ----------------------------------------------------
        # RUN LANGGRAPH
        # ----------------------------------------------------

        try:

            print(
                Fore.MAGENTA
                + "⚙️ Processing your question..."
            )


            result = app.invoke(
                {
                    "programme": student_programme,

                    "messages": [
                        (
                            "human",
                            user_query
                        )
                    ]
                }
            )


            answer = result[
                "messages"
            ][-1].content


            print(
                Fore.CYAN
                + "\n"
                + "-" * 65
            )


            print(
                Fore.GREEN
                + "🤖 Assistant:"
            )


            print(
                Fore.WHITE
                + answer
            )


            print(
                Fore.CYAN
                + "-" * 65
            )


        except Exception as e:

            print(
                Fore.RED
                + "\n❌ Something went wrong."
            )

            print(
                Fore.RED
                + f"Error: {e}"
            )