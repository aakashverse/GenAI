from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from typing import Annotated

# make chatbot graph's state
class ChatState(BaseModel):
    messages: Annotated[list, add_messages]

# llm
llm = ChatGroq(model="openai/gpt-oss-20b")

# make graph's node
def chatBotNode(state: ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

# memory
memory = InMemorySaver()

# make graph
graph = StateGraph(ChatState)
graph.add_node("chatBot", chatBotNode)
graph.add_edge(START, "chatBot")
graph.add_edge("chatBot", END)

graph = graph.compile(checkpointer=memory)

# invoke graph
while True:
    query = input("User: ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("Good bye, Take care :>")
        break

    res = graph.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":"my-bot-1"}}
        )
    
    ans = res["messages"][-1].content
    print("AI: ", ans)
    