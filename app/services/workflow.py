from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState

from .llm import generate_query_or_respond, qdrant_retrieve
from .grade_docs import rewrite_question, grade_documents, generate_answer


workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([qdrant_retrieve]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

def run_workflow(question: str, k: int = 5) -> str:
    """
    Run the agentic RAG workflow for a given question.
    
    Args:
        question: User's travel-related question
        k: Number of documents to retrieve
        
    Returns:
        Final answer as a string
    """
    state = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    }
    
    result = graph.invoke(state)
    
    # Extract the final answer
    last_message = result["messages"][-1]
    
    # Handle different message formats
    if hasattr(last_message, "content"):
        content = last_message.content
        if isinstance(content, list):
            # Extract text from list format
            text_content = next((item["text"] for item in content if item.get("type") == "text"), None)
            return text_content or str(content)
        return content
    
    return str(last_message)
