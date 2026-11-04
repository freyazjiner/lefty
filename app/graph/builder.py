from langgraph.graph import END, StateGraph

from app.graph.nodes.chitchat import handle_chitchat
from app.graph.nodes.clarify import clarify_shop
from app.graph.nodes.compare import compare_options
from app.graph.nodes.parse import parse_request
from app.graph.nodes.router import route_after_parse
from app.graph.nodes.search import search_catalog
from app.graph.nodes.shop_reply import generate_shop_reply
from app.graph.state import AgentState
from app.observability.timing import timed_node

_agent_graph = None


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse", timed_node("parse")(parse_request))
    workflow.add_node("clarify", timed_node("clarify")(clarify_shop))
    workflow.add_node("chitchat", timed_node("chitchat")(handle_chitchat))
    workflow.add_node("search", timed_node("search")(search_catalog))
    workflow.add_node("compare", timed_node("compare")(compare_options))
    workflow.add_node("reply", timed_node("reply")(generate_shop_reply))

    workflow.set_entry_point("parse")

    workflow.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "clarify": "clarify",
            "chitchat": "chitchat",
            "search": "search",
        },
    )

    workflow.add_edge("clarify", END)
    workflow.add_edge("chitchat", END)
    workflow.add_edge("search", "compare")
    workflow.add_edge("compare", "reply")
    workflow.add_edge("reply", END)

    return workflow.compile()


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
