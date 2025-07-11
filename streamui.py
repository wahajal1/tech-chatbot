import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.memory import ConversationBufferMemory

@st.cache_resource
def load_agent():
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def dictionary_tool_fn(query: str) -> str:
        simple_dict = {
            "python": "Python is a high-level, general-purpose programming language...",
            "AI": "AI is the field of computer science that focuses on simulating intelligence...",
            "data science": "Data science is the process of extracting insights from data...",
        }
        return simple_dict.get(query, "Unkown")

    dictionary_tool = Tool(
        name="dictionary_tool",
        func=dictionary_tool_fn,
        description="Use this tool to define technical terms related to programming."
    )

    tools = [wiki_tool, dictionary_tool]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        memory=memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors = True
    )

    return agent, memory


agent, memory = load_agent()


st.title("💬 Tech ChatBot")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Write your question here...")

if user_input:
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("⏳ Thinking..."):
         response = agent.run(user_input)
         try:
        
          answer = str(response).strip()
          if answer.lower() in ["none", "undefined", ""]:
           answer = "🤖 Sorry, I couldn't generate a clear answer."
         except Exception:
           answer = "🤖 Unexpected error in formatting the response."

  
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
