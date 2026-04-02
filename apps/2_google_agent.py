from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a search agent & can search for any queries on google.",
    checkpointer=memory
)

while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Good Bye :)")
        break

    response = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable":{"thread_id":"akash01"}},
        )
    print("AI: ", response['messages'][-1].content)