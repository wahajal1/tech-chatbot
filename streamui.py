import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.memory import ConversationBufferMemory

os.environ["GOOGLE_API_KEY"] ="AIzaSyDxcmxnwx9cj5G7hnnLDg63XmwbubzAymo"

@st.cache_resource
def load_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key ="AIzaSyDxcmxnwx9cj5G7hnnLDg63XmwbubzAymo" )

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

# إعداد سجل المحادثة
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📥 واجهة إدخال المستخدم
user_input = st.chat_input("Write your question here...")

# 📤 تنفيذ الرد إذا فيه إدخال
if user_input:
    # أضف رسالة المستخدم إلى السجل
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("⏳ Thinking..."):
        try:
            response = agent.run(user_input)

            # ✅ التأكد إن الناتج ما هو undefined أو None
            if not response or str(response).strip().lower() in ["none", "undefined"]:
                answer = "🤖 Sorry, I couldn't understand that."
            else:
                answer = str(response).strip()

        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

    # أضف رد البوت إلى السجل
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ✅ عرض سجل المحادثة بعد الرد (يمنع التكرار وظهور undefined)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
