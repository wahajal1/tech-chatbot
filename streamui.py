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

# chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# showing the last message
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
     st.markdown(msg["content"])

#user input
user_input = st.chat_input("Write your question here...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("⏳ Loading..."):
            try:
                response = agent.run(user_input)
                res = str(response)
                st.markdown(res)
                st.session_state.chat_history.append({"role": "assistant", "content": res})

            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                st.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})تشاهد مرات الـChatBot عندما يجيب على أسئلة ويجيب عليها ولكن يضع تحتها Undefined وليس على كل الأسئلة ثاني شيء عندما يجيب على أسئلة ويجيب عليها يتكرر مرتين ثم يختفي فكيف أعدل هذه الأشياء؟ هذا كودي

ChatGPT said:
المشكلة التي تواجهك تنقسم إلى نقطتين رئيسيتين كما ذكرت:

✅ أولًا: ظهور undefined تحت بعض الإجابات
السبب:
السبب في هذا هو أن بعض الأدوات (مثل WikipediaQueryRun) أو دالة dictionary_tool_fn قد ترجع None أو قيمة غير متوقعة، مما يؤدي إلى عرض undefined في الواجهة (وخصوصًا إذا كانت نتيجة الرد None أو Unknown أو نص غير قابل للعرض بشكل صحيح).

الحل:
قم بإضافة معالجة للنتيجة قبل إظهارها. وتأكد من أن النتيجة دائمًا نص قابل للعرض. أضف هذا السطر قبل st.markdown(res):

python
Copy
Edit
res = str(response).strip() if response else "🤖 I'm not sure how to respond to that."
✅ ثانيًا: تكرار الرد مرتين ثم اختفاؤه
السبب:
أنت تقوم بطباعة الرد في st.chat_message("assistant") وتخزنه في session_state.chat_history، ثم تعيد طباعته مرة أخرى في حلقة for التي تعرض كل الرسائل.

الحل:
تحتاج إلى تأخير تحديث سجل المحادثة حتى بعد عرض الرسالة الجديدة. أو تضيف شرطًا لتجنب تكرارها فورًا.

التعديل المقترح:
ضع هذا الكود في مكان واحد فقط لعرض الرسائل، وأزل التكرار. أفضل تعديل يكون كالتالي:

✅ التعديل الكامل على الجزء الخاص بـ user input:
python
Copy
Edit
user_input = st.chat_input("Write your question here...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        with st.spinner("⏳ Loading..."):
            response = agent.run(user_input)
            res = str(response).strip() if response else "🤖 I'm not sure how to respond to that."

    except Exception as e:
        res = f"⚠️ Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": res})

# إعادة عرض الرسائل فقط بعد التحديث
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
✅ تحسين إضافي (اختياري):
في dictionary_tool_fn، غير من "Unkown" إلى "Unknown term. Please try another word." أو شيء أكثر وضوحًا.

python
Copy
Edit
return simple_dict.get(query.lower(), "🤖 Unknown term. Please try another word.")
هل ترغبين أن أرسل لك نسخة كاملة من الكود بعد التعديل؟








You said:
لا، بس أنت ما فهمتني، النقطة الثانية تمام، لكن النقطة الأولى هو يجاوب مو إنه ما يعرف الجواب، لا، يجاوب بس غصب تطلع لي تحتها Undefined، كيف أشيل هذي Undefined؟ يعني هو يعرف الجواب، قاعد يجاوبني


ChatGPT said:
شكرًا على التوضيح! الآن فهمت قصدك تمامًا 🔍

أنت تقصد إن البوت يجيب بشكل صحيح (مثلاً يعطيك تعريف Java)، لكن بعد الرد مباشرة يظهر سطر إضافي مكتوب فيه undefined، وهذا السطر ما له داعي.

✅ السبب الحقيقي:
هذه undefined لا تأتي من الباكند (الكود نفسه)، بل من واجهة Streamlit – تحديدًا من st.chat_input().

عندما تستخدمي st.chat_input() داخل الصفحة، يتم تلقائيًا محاولة إرسال القيمة حتى لو ما في رد، أو يتم إعادة تحديث بسيط بعد ظهور الرد، مما يؤدي أحيانًا إلى ظهور سطر فارغ (أو undefined) في صندوق المحادثة.

✅ الحل العملي:
بدل ما تطبعي الرد مباشرة باستخدام st.chat_message("assistant")، ثم تسجليه بعدها، اعكسي الترتيب، واجعلي جميع الرسائل تُعرض فقط من st.session_state.chat_history.

🎯 الخطوات لتجنب undefined:
1. أزيلي هذا الجزء 👇 لأنه يعرض الرد مباشرة:
python
Copy
Edit
    with st.chat_message("assistant"):
        with st.spinner("⏳ Loading..."):
            try:
                response = agent.run(user_input)
                res = str(response)
                st.markdown(res)
                st.session_state.chat_history.append({"role": "assistant", "content": res})
2. استبدليه بهذا الكود:
python
Copy
Edit
    with st.spinner("⏳ Loading..."):
        try:
            response = agent.run(user_input)
            res = str(response).strip() if response else "🤖 I'm not sure how to respond to that."
        except Exception as e:
            res = f"⚠️ Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": res})
3. أبقي فقط عرض الرسائل من التاريخ:
python
Copy
Edit
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
✅ ملخص:
سبب ظهور undefined: غالبًا من Streamlit بعد الرد بسبب عرض مزدوج أو فوري.

الحل: لا تطبع الرد فورًا، بل فقط بعد إضافته إلى chat_history.

ضعي كل عرض الرسائل داخل for msg in st.session_state.chat_history.

لو حابة أرسل لك النسخة المعدلة كلها جاهزة بدون undefined، أخبريني فقط.








You said:
مافهمت ارسل حل النقطتين سوا


ChatGPT said:
تمام واهج! هذا هو الحل الكامل للكود حقك بعد تعديل النقطتين:

✅ المشكلة الأولى: ظهور undefined
🔧 الحل: لا تطبع الرد فورًا باستخدام st.chat_message("assistant")، فقط خزنه ثم اطبعه من st.session_state.chat_history.

✅ المشكلة الثانية: تكرار الرد مرتين ثم اختفاؤه
🔧 الحل: اجعل كل الطباعة تتم فقط من خلال حلقة for بعد تحديث chat_history.

✅ الكود الكامل بعد التعديل:
python
Copy
Edit
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.memory import ConversationBufferMemory

os.environ["GOOGLE_API_KEY"] = "AIzaSyDxcmxnwx9cj5G7hnnLDg63XmwbubzAymo"

@st.cache_resource
def load_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key="AIzaSyDxcmxnwx9cj5G7hnnLDg63XmwbubzAymo"
    )

    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def dictionary_tool_fn(query: str) -> str:
        simple_dict = {
            "python": "Python is a high-level, general-purpose programming language...",
            "AI": "AI is the field of computer science that focuses on simulating intelligence...",
            "data science": "Data science is the process of extracting insights from data...",
        }
        return simple_dict.get(query.lower(), "🤖 Unknown term. Please try another word.")

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
        handle_parsing_errors=True
    )

    return agent, memory


agent, memory = load_agent()

st.title("💬 Tech ChatBot")

# تهيئة سجل المحادثة
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ عرض كل الرسائل فقط من سجل المحادثة (لتجنب undefined والتكرار)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 📩 استقبال رسالة المستخدم
user_input = st.chat_input("Write your question here...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("⏳ Loading..."):
        try:
            response = agent.run(user_input)
            res = str(response).strip() if response else "🤖 I'm not sure how to respond to that."
        except Exception as e:
            res = f"⚠️ Error: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": res})









