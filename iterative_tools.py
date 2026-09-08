import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# Tools
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# LLMs
writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

# State Building
class State(TypedDict):
    topic: str 
    messages: Annotated[list, add_messages]
    draft: str 
    review_feedback: str
    is_approved: bool 
    attempt: int

# Prompts
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

# Nodes
def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call search tools first."""
    attempt = state.get("attempt", 0) + 1 
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: '{topic}'. "
            f"If you need current information, search the web first."
        )
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n"
            f"Here is the reviewer's feedback:\n{previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentioned. "
            f"Do not repeat the same mistakes."
        )
    
    # Construct conversation history
    current_messages = list(state.get("messages", []))
    prompt_messages = [("system", WRITER_SYSTEM_PROMPT)] + current_messages + [("human", user_message)]
    
    response = writer_llm_with_tools.invoke(prompt_messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt
    }

tool_node = ToolNode(tools)

def extract_draft_node(state: State) -> dict:
    """Extracts final writer message as the active draft."""
    last_message = state['messages'][-1]
    draft = last_message.content 
    print(f"\n--- Generated Post (Attempt {state.get('attempt')}) ---\n{draft}\n")
    return {"draft": draft}

def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    prompt = f"Review this LinkedIn post draft:\n\n{draft}\n\nProvide your verdict and feedback."
    response = reviewer_llm.invoke([("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)])
    review_text = response.content.strip()
    
    # Split check for verdict
    is_approved = "APPROVED" in review_text.split("FEEDBACK:")[0].upper()

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]\n")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved,
    }

# Conditional Edges
def should_use_tool(state: State):
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"

def should_stop_looping(state: State):
    if state['is_approved']:
        print("Post has been approved.\n")
        return END
    if state['attempt'] >= 3:
        print("Reached maximum attempts limit.\n")
        return END 
    return "writer"

# Graph Construction
graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")

# Writer routes either to tool execution or draft extraction
graph.add_conditional_edges("writer", should_use_tool)

# Tools MUST route back to the writer to synthesize results into a draft
graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "reviewer")

# Reviewer routes either to retry (writer) or completion (END)
graph.add_conditional_edges("reviewer", should_stop_looping)

app = graph.compile()

# Execution
if __name__ == "__main__":
    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

    if not topic:
        print("\nNo topic given. Exiting.")
    else:
        print("\nStarting generation...\n")

        initial_state = {
            "topic": topic,
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }

        final_state = app.invoke(initial_state)

        print("\n" + "=" * 55)
        print("FINAL LINKEDIN POST")
        print("=" * 55)
        print(final_state["draft"])
        print("=" * 55)
        print(f"Total attempts: {final_state['attempt']}")
        print(f"Approved: {final_state['is_approved']}")