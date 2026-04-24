# core/graph.py
from langgraph.graph import StateGraph, START, END
from core.state import MarketMindState

# Import all 6 of our nodes
from core.agents.price_agent import price_node
from core.agents.sentiment_agent import sentiment_node
from core.agents.macro_agent import macro_node
from core.agents.onchain_agent import onchain_node
from core.agents.risk_agent import risk_node
from core.agents.synthesis_agent import synthesis_node

# 1. Initialize the Graph with our global State
builder = StateGraph(MarketMindState)

# 2. Register all the nodes (the "rooms" in our office)
builder.add_node("price_agent", price_node)
builder.add_node("sentiment_agent", sentiment_node)
builder.add_node("macro_agent", macro_node)
builder.add_node("onchain_agent", onchain_node)
builder.add_node("risk_agent", risk_node)
builder.add_node("synthesis_agent", synthesis_node)

# 3. THE FAN-OUT (Parallel Execution)
# By connecting START to all 5 agents simultaneously, LangGraph knows 
# to fire them all at the exact same time in parallel.
builder.add_edge(START, "price_agent")
builder.add_edge(START, "sentiment_agent")
builder.add_edge(START, "macro_agent")
builder.add_edge(START, "onchain_agent")
builder.add_edge(START, "risk_agent")

# 4. THE FAN-IN (Synthesis)
# Connect all 5 specialists to the Synthesis Boss. 
# LangGraph will automatically wait until ALL 5 are finished before running Synthesis.
builder.add_edge("price_agent", "synthesis_agent")
builder.add_edge("sentiment_agent", "synthesis_agent")
builder.add_edge("macro_agent", "synthesis_agent")
builder.add_edge("onchain_agent", "synthesis_agent")
builder.add_edge("risk_agent", "synthesis_agent")

# 5. Finish the process
builder.add_edge("synthesis_agent", END)

# 6. Compile the graph into an executable application
marketmind_app = builder.compile()