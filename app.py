import streamlit as st
import os
import docx
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

# --- 常量定义 ---
FAISS_INDEX_PATH = "faiss_index"

# --- UI界面设计 ---
st.set_page_config(page_title="现代汉语语法三百问智思体检索系统", layout="wide")
st.title("📖 现代汉语语法三百问智思体检索系统")

# ====================================================================
# 核心修改：使用CSS注入来创建固定在主页面底部的页脚
# ====================================================================
footer_css = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: var(--background-color); /* 适应Streamlit主题背景色 */
    color: var(--text-color); /* 适应Streamlit主题文字颜色 */
    text-align: center;
    padding: 0.5rem;
    /* 为侧边栏留出空间，避免页脚文字被侧边栏遮挡 */
    padding-left: 21rem; 
    z-index: 99;
}
/* 当屏幕宽度较小时（例如手机），侧边栏会收起，此时不再需要左边距 */
@media (max-width: 768px) {
    .footer {
        padding-left: 0.5rem;
    }
}
</style>
"""

footer_html = """
<div class="footer">
    <p>Copyright © 2025-&nbsp;&nbsp;&nbsp;版权所有：华中师范大学沈威&nbsp;&nbsp;&nbsp;邮箱：sw@ccnu.edu.cn</p>
</div>
"""
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)

# --- API Key及模型配置 ---
with st.sidebar:
    st.header("⚙️ 系统核心设置")
    api_key_input = st.text_input(
        "请输入您的Google API Key:",
        type="password",
        help="系统也会自动检测'GOOGLE_API_KEY'环境变量。"
    )
    # 此处的版权信息已被移出
    final_api_key = api_key_input if api_key_input else os.getenv("GOOGLE_API_KEY")

if not final_api_key:
    st.error("❌ 请在左侧边栏输入您的Google API Key或设置'GOOGLE_API_KEY'环境变量。")
    st.stop()

try:
    genai.configure(api_key=final_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=final_api_key,
        task_type="RETRIEVAL_QUERY"
    )

except Exception as e:
    st.error(f"API Key配置或模型初始化失败，请检查您的Key是否正确。错误信息: {e}")
    st.stop()


# --- 知识库处理核心函数 (优化版，包含分批处理和重试) ---
def process_and_save_document(uploaded_file):
    with st.spinner("首次初始化：正在处理文档并构建向量知识库..."):
        try:
            doc = docx.Document(uploaded_file)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text])

            if not full_text.strip():
                st.error("上传的文档为空或不包含任何文本，请检查后重新上传。")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)

            if not chunks:
                st.error("文档切分后未产生任何文本块，请检查文档内容。")
                return

            batch_size = 100

            st.info(f"检测到 {len(chunks)} 个文本块。开始分批构建知识库...")
            vector_store = FAISS.from_texts(
                texts=chunks[:batch_size],
                embedding=embeddings
            )
            st.success(f"已成功处理前 {min(batch_size, len(chunks))}/{len(chunks)} 个文本块。")

            for i in range(batch_size, len(chunks), batch_size):
                retries = 3
                while retries > 0:
                    try:
                        batch = chunks[i:i + batch_size]
                        vector_store.add_texts(texts=batch)
                        st.success(f"已成功处理 {min(i + batch_size, len(chunks))}/{len(chunks)} 个文本块。")
                        time.sleep(1)
                        break
                    except Exception as e:
                        retries -= 1
                        st.warning(f"处理批次时出错: {e}。剩余重试次数: {retries}。正在等待后重试...")
                        time.sleep(5)
                if retries == 0:
                    st.error(f"批次 {i // batch_size + 1} 重试多次后仍然失败，知识库构建中止。")
                    raise Exception("Failed to process document in batches.")

            vector_store.save_local(FAISS_INDEX_PATH)
            st.info("所有文本块处理完毕，向量知识库已成功保存！")

        except Exception as e:
            st.error(f"处理文档并创建知识库时出错: {e}")
            if os.path.exists(FAISS_INDEX_PATH):
                import shutil
                shutil.rmtree(FAISS_INDEX_PATH)


@st.cache_resource(show_spinner="正在加载向量知识库...")
def load_vector_store():
    try:
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
    # 为了给固定的页脚留出空间，可以在聊天容器底部增加一个空白区域
    st.container().empty()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 在聊天界面的最后，增加一个与页脚等高的空白div，防止最后一条消息被页脚遮挡
    if len(st.session_state.messages) > 0:
        st.markdown("<div style='padding-bottom: 4rem;'></div>", unsafe_allow_html=True)

    if user_question := st.chat_input("请就现代汉语语法进行提问..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 智思体正在检索与思考中..."):
                docs = vector_store.similarity_search(user_question, k=3)

                if not docs:
                    response = "本知识库里不包含这个问题。"
                else:
                    question_prompt_template = """
                    请严格根据以下提供的上下文信息来回答问题。
                    确保你的回答完全基于这些信息，不要添加任何外部知识。
                    如果根据上下文无法回答问题，请指明信息不足。
                    上下文:
                    {context}
                    问题:
                    {question}
                    回答:
                    """
                    QUESTION_PROMPT = PromptTemplate(
                        template=question_prompt_template, input_variables=["context", "question"]
                    )

                    combine_prompt_template = """
                    现有多个从文档中抽取的答案片段，请对这些片段进行全面和详细的整合与总结，形成一个最终的、流畅且唯一的回答。
                    请专注于整合信息，不要重复“根据提供的上下文”之类的话语。
                    如果你认为这些片段累计起来仍无法回答原始问题，请直接说：“本知识库里不包含这个问题。”
                    抽取的答案片段:
                    {summaries}
                    请根据以上片段，给出最终的详细回答:
                    """
                    COMBINE_PROMPT = PromptTemplate(
                        template=combine_prompt_template, input_variables=["summaries"]
                    )

                    model = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=final_api_key,
                        temperature=0.3
                    )

                    chain = load_qa_chain(
                        llm=model,
                        chain_type="map_reduce",
                        question_prompt=QUESTION_PROMPT,
                        combine_prompt=COMBINE_PROMPT
                    )

                    response_obj = chain.invoke(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    response = response_obj['output_text']

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                # 响应后立即重新运行，以确保底部的空白填充物被渲染
                st.rerun()
