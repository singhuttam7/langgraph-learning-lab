import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)

load_dotenv()

#Step 1 - Building the RAG retrievers
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def build_retriever(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)

    chunks =  splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k":4})

academic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

#Step2 = State

class State(TypedDict):
    programme: str
    messages : Annotated[list, add_messages]
    query_type: str
    retrieved_context: str

#Step 3 - Nodes generation

def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
        "Do not use Markdown formatting. Do not use #, *, **, bullet points, tables, or backticks. "
        "Return plain text only with simple paragraphs."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
       category = "academic"
    elif "fee" in category:
       category = "fee"
    else:
       category = "general"

    return {"query_type": category}

def academic_rag_node(state: State)-> dict:
   """Retriever relevant chunks from the academics handbook."""
   query = state["messages"][-1].content
   docs = academic_retriever.invoke(query)
   context = "\n\n".join([doc.page_content for doc in docs])

   return {"retrieved_context": context}

def fee_rag_node(state: State)-> dict:
   """Retriever relevant chunks from the fee structuring PDF."""
   query = state["messages"][-1].content
   docs = fee_retriever.invoke(query)
   context = "\n\n".join([doc.page_content for doc in docs])
   return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: State)-> dict:
   """Generates the final answer, personalized using the student's programme."""
   query = state["messages"][-1].content
   programme = state.get("programme", "unknown")
   context =  state["retrieved_context"]

   if context == "NO_RETRIEVAL_NEEDED":
      prompt =(
         f"You are a friendly college assistant talking to a {programme} student. "
         f"Answer this question using your own general knowledge:\n\n{query}"
      )
   else:
      prompt =(
          f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
            "Do not use Markdown formatting. Do not use #, *, **, bullet points, tables, or backticks. "
            "Return plain text only with simple paragraphs." 
      )

   response = llm.invoke(prompt)
   return {"messages": [("ai", response.content.strip())]}

#Step 4 - Router function

def route_query(state: State):
   if state['query_type'] == 'academic':
      return "academic_rag"
   elif state['query_type'] == "fee":
      return "fee_rag"

   else:
      return "general"


#step 5 - Building the graph

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

#edges

graph.add_edge(START, "classifier")
graph.add_conditional_edges(
   "classifier", route_query
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()
# ============================================================
# STEP 6 - RUN THE APPLICATION
# ============================================================

def print_banner():
    print("\n" + Fore.CYAN + "=" * 65)
    print(Fore.CYAN + "           🎓 COLLEGE AI ASSISTANT")
    print(Fore.CYAN + "=" * 65)
    print(Fore.WHITE + "      Intelligent RAG-powered Student Assistant")
    print(Fore.CYAN + "-" * 65)


def print_programme_menu():
    print(Fore.YELLOW + "\n📚 SELECT YOUR PROGRAMME")
    print(Fore.WHITE + "-" * 40)
    print(Fore.GREEN + "  [1] " + Fore.WHITE + "BCA")
    print(Fore.GREEN + "  [2] " + Fore.WHITE + "BBA")
    print(Fore.GREEN + "  [3] " + Fore.WHITE + "B.Com (H)")
    print(Fore.WHITE + "-" * 40)


# Start application
print_banner()

print_programme_menu()

choice = input(Fore.YELLOW + "\nEnter your choice (1/2/3): " + Style.RESET_ALL)

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}

student_programme = programme_map.get(choice)

# Validate programme choice
if student_programme is None:
    print(Fore.RED + "\n❌ Invalid choice! Defaulting to BCA.")
    student_programme = "BCA"

print(
    Fore.GREEN
    + f"\n✓ Programme selected: {student_programme}"
)

print(Fore.CYAN + "=" * 65)

print(
    Fore.WHITE
    + "\n💡 You can ask questions about academics, fees, "
      "attendance, exams, grading, etc."
)

print(
    Fore.YELLOW
    + "   Type 'exit' or 'quit' to close the assistant."
)

print(Fore.CYAN + "=" * 65)


# Chat loop
while True:

    print()

    user_query = input(
        Fore.BLUE + "👤 You: " + Style.RESET_ALL
    )

    # Exit condition
    if user_query.lower().strip() in ["exit", "quit"]:

        print(Fore.CYAN + "\n" + "=" * 65)
        print(Fore.GREEN + "👋 Thank you for using College AI Assistant!")
        print(Fore.CYAN + "=" * 65 + "\n")

        break

    # Ignore empty input
    if not user_query.strip():
        print(Fore.YELLOW + "⚠️ Please enter a question.")
        continue

    try:

        print(
            Fore.MAGENTA
            + "⚙️  Processing your question..."
        )

        result = app.invoke({
            "programme": student_programme,
            "messages": [("human", user_query)]
        })

        answer = result["messages"][-1].content

        print(Fore.CYAN + "\n" + "-" * 65)

        print(
            Fore.GREEN
            + "🤖 Assistant:"
        )

        print(Fore.WHITE + answer)

        print(Fore.CYAN + "-" * 65)

    except Exception as e:

        print(Fore.RED + "\n❌ Something went wrong.")
        print(Fore.RED + f"Error: {e}")

