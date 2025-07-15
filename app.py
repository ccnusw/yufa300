import streamlit as st
import os
import docx
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
# 移除不再需要的 'import google.generativeai as genai'

# --- 常量定义 ---
FAISS_INDEX_PATH = "faiss_index"

# --- UI界面设计 ---
st.set_page_config(page_title="现代汉语语法三百问智思体检索系统", layout="wide")
st.title("📖 现代汉语语法三百问智思体检索系统")

# --- API Key及模型配置 ---
with st.sidebar:
    st.header("⚙️ 系统核心设置")
    api_key_input = st.text_input(
        "请输入您的Google API Key:",
        type="password",
        help="系统也会自动检测'GOOGLE_API_KEY'环境变量。"
    )

    final_api_key = api_key_input if api_key_input else os.getenv("GOOGLE_API_KEY")

if not final_api_key:
    st.error("❌ 请在左侧边栏输入您的Google API Key或设置'GOOGLE_API_KEY'环境变量。")
    st.stop()

# --- 初始化Embeddings模型 ---
# 将API Key直接传递给构造函数，确保其被正确使用
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_QUERY",
        google_api_key=final_api_key
    )
except Exception as e:
    st.error(f"Embeddings模型初始化失败，请检查您的API Key。错误信息: {e}")
    st.stop()


# --- 知识库处理核心函数 ---
def process_and_save_document(uploaded_file):
    """处理上传的Word文档，创建并保存向量索引"""
    with st.spinner("首次初始化：正在处理文档并构建向量知识库..."):
        try:
            doc = docx.Document(uploaded_file)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text])

            if not full_text.strip():
                st.error("上传的文档为空或不包含任何文本，请检查后重新上传。")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)

            # 使用已经初始化好的embeddings模型
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)

        except Exception as e:
            st.error(f"处理文档并创建知识库时出错: {e}")
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)


@st.cache_resource(show_spinner="正在加载向量知识库...")
def load_vector_store():
    """从本地加载向量索引"""
    try:
        # 加载时也需要传入embeddings实例
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"加载本地向量知识库失败: {e}. 如果您刚刚上传了新文件，请刷新页面。")
        return None


# --- 主逻辑 ---
vector_store = None

if os.path.exists(FAISS_INDEX_PATH):
    st.sidebar.success("✅ 向量知识库已加载，系统准备就绪！")
    vector_store = load_vector_store()
else:
    st.sidebar.warning("⚠️ 系统中未找到向量知识库。")
    st.sidebar.markdown("---")
    st.sidebar.header("首次初始化")
    uploaded_file = st.sidebar.file_uploader(
        "请上传初始的Word知识库文档 (.docx)",
        type="docx"
    )

    if uploaded_file is not None:
        process_and_save_document(uploaded_file)
        st.sidebar.success("知识库构建完成并已保存！页面将自动刷新以加载。")
        st.rerun()
    else:
        st.info("请“管理员”在左侧边栏上传一个Word文档来构建系统知识库。")
        st.stop()

# --- 聊天界面 ---
if vector_store:
    st.markdown("""
    欢迎使用本系统！知识库已准备好，您现在可以在下方的聊天框中开始提问。
    """)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("请就现代汉语语法进行提问..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 智思体正在检索与思考中..."):
                docs = vector_store.similarity_search(user_question, k=3)

                if not docs:
                    response = "抱歉，在知识库中没有找到与您问题相关的信息。"
                else:
                    prompt_template = """
                    请严格根据以下提供的上下文信息来回答问题。
                    确保你的回答完全基于这些信息，不要添加任何外部知识。
                    如果根据上下文无法回答问题，请直接说：“抱歉，在知识库中没有找到与您问题相关的信息。”

                    上下文:
                    {context}

                    问题:
                    {question}

                    回答:
                    """
                    prompt = PromptTemplate(
                        template=prompt_template, input_variables=["context", "question"]
                    )
                    # 显式传递API Key并使用新的Flash模型
                    model = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-latest",
                        temperature=0.3,
                        google_api_key=final_api_key
                    )
                    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

                    response_obj = chain.invoke(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    response = response_obj['output_text']

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
