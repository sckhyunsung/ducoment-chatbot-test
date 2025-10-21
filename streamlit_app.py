import os
import time
import tempfile
import email
from email import policy
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# =============================================================
# Environment tweaks
# =============================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================
# LangChain-compatible imports with robust fallbacks
# =============================================================
# Text Splitter (new package -> newer LC namespace -> old LC namespace)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # newest
    _SPLITTER_SRC = "langchain_text_splitters"
except Exception:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter  # newer LC
        _SPLITTER_SRC = "langchain.text_splitters"
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # old LC
        _SPLITTER_SRC = "langchain.text_splitter"

# VectorStore FAISS (community -> old)
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    from langchain.vectorstores import FAISS  # very old fallback

# Embeddings & Chat LLM (langchain_openai)
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except Exception as e:
    st.error("❌ 'langchain_openai'가 설치되어 있지 않습니다. requirements.txt에 'langchain-openai>=0.2.0'을 추가하세요.")
    raise

# Document loaders (prefer community; fallback to old; allow None)
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
    )
except Exception:
    try:
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            UnstructuredWordDocumentLoader,
            UnstructuredPowerPointLoader,
        )
    except Exception:
        PyPDFLoader = None
        TextLoader = None
        UnstructuredWordDocumentLoader = None
        UnstructuredPowerPointLoader = None

# Prompts
from langchain.prompts import ChatPromptTemplate

# Tool & Agent (newer paths first)
try:
    from langchain.tools import Tool
except Exception:
    try:
        from langchain.agents import Tool  # very old
    except Exception:
        Tool = None

try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    _AGENTS_AVAILABLE = True
except Exception:
    # Fallback: we'll run a manual RAG pipeline (no Agent)
    create_tool_calling_agent = None
    AgentExecutor = None
    _AGENTS_AVAILABLE = False

# =============================================================
# Utility: simple banner for current import modes (helps debugging)
# =============================================================
def _debug_imports_banner():
    st.caption(
        f"🔧 Imports — Splitter: `{_SPLITTER_SRC}`, Agent: "
        + ("ON (create_tool_calling_agent)" if _AGENTS_AVAILABLE else "OFF (manual RAG)")
    )

# =============================================================
# Document loading helpers
# =============================================================
class SimpleDocument:
    """Lightweight fallback if LC Document import path ever changes."""
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Prefer official LC Document class, but fall back if needed
try:
    from langchain_core.documents import Document as LCDocument
except Exception:
    try:
        from langchain.schema import Document as LCDocument
    except Exception:
        LCDocument = None

DocumentCls = LCDocument if LCDocument is not None else SimpleDocument


def _read_txt_fallback(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="cp949", errors="ignore") as f:
            return f.read()


# -------------------------------------------------------------
# Load various file types into a list[Document]
# -------------------------------------------------------------
def load_documents(uploaded_files) -> List[DocumentCls]:
    all_documents: List[DocumentCls] = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📂 파일 처리 중: {uploaded_file.name} ({idx+1}/{total_files})")
        progress_bar.progress((idx + 1) / total_files)

        file_extension = uploaded_file.name.split(".")[-1].lower()

        # Make a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if file_extension == "pdf":
                if PyPDFLoader is None:
                    st.warning("⚠️ PyPDFLoader를 사용할 수 없습니다. 'langchain_community' 또는 호환 버전을 설치하세요.")
                    continue
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

            elif file_extension == "txt":
                # Use LC TextLoader if available, otherwise simple read
                if TextLoader is not None:
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                    documents = loader.load()
                    for d in documents:
                        d.metadata["source"] = uploaded_file.name
                else:
                    text = _read_txt_fallback(tmp_file_path)
                    documents = [DocumentCls(page_content=text, metadata={"source": uploaded_file.name, "type": "Text"})]

            elif file_extension == "csv":
                df = pd.read_csv(tmp_file_path)
                text = df.to_string()
                documents = [DocumentCls(page_content=text, metadata={"source": uploaded_file.name, "type": "CSV"})]

            elif file_extension in ["xlsx", "xls"]:
                excel_file = pd.ExcelFile(tmp_file_path)
                documents = []
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(tmp_file_path, sheet_name=sheet_name)
                    text = f"Sheet: {sheet_name}\n\n{df.to_string()}"
                    documents.append(
                        DocumentCls(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "sheet": sheet_name, "type": "Excel"},
                        )
                    )

            elif file_extension in ["docx", "doc"]:
                if UnstructuredWordDocumentLoader is None:
                    st.warning("⚠️ Word 로더가 없음: 'unstructured' 관련 의존성을 설치하세요 (또는 Word를 다른 포맷으로 변환).")
                    continue
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
                documents = loader.load()
                for d in documents:
                    d.metadata["source"] = uploaded_file.name
                    d.metadata["type"] = "Word"

            elif file_extension in ["pptx", "ppt"]:
                if UnstructuredPowerPointLoader is None:
                    st.warning("⚠️ PowerPoint 로더가 없음: 'unstructured' 관련 의존성을 설치하세요 (또는 PPT를 PDF로 저장 후 업로드).")
                    continue
                loader = UnstructuredPowerPointLoader(tmp_file_path)
                documents = loader.load()
                for d in documents:
                    d.metadata["source"] = uploaded_file.name
                    d.metadata["type"] = "PowerPoint"

            elif file_extension in ["mhtml", "mht"]:
                try:
                    with open(tmp_file_path, "rb") as f:
                        msg = email.message_from_binary_file(f, policy=policy.default)

                    text_parts = []
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type in ["text/html", "text/plain"]:
                            payload = part.get_payload(decode=True)
                            if not payload:
                                continue
                            charset = part.get_content_charset() or "utf-8"
                            try:
                                text = payload.decode(charset, errors="ignore")
                            except Exception:
                                text = payload.decode("utf-8", errors="ignore")
                            if content_type == "text/html":
                                soup = BeautifulSoup(text, "html.parser")
                                text = soup.get_text(separator="\n", strip=True)
                            text_parts.append(text)

                    if text_parts:
                        combined_text = "\n\n".join(text_parts)
                        documents = [DocumentCls(page_content=combined_text, metadata={"source": uploaded_file.name, "type": "MHTML"})]
                    else:
                        st.warning(f"⚠️ MHTML 텍스트 추출 실패: {uploaded_file.name}")
                        continue
                except Exception as e:
                    st.error(f"❌ MHTML 처리 오류: {uploaded_file.name} - {str(e)}")
                    continue

            else:
                st.warning(f"⚠️ 지원하지 않는 파일 형식: {uploaded_file.name}")
                continue

            # Normalize metadata for LC vs SimpleDocument
            norm_docs = []
            for d in documents:
                # Some loaders may yield dict-like objects in very old versions
                if hasattr(d, "page_content"):
                    content = d.page_content
                    meta = getattr(d, "metadata", {}) or {}
                else:
                    try:
                        content = d["page_content"]
                        meta = d.get("metadata", {}) or {}
                    except Exception:
                        continue
                norm_docs.append(DocumentCls(page_content=content, metadata=meta))

            all_documents.extend(norm_docs)

        except Exception as e:
            st.error(f"❌ 파일 로드 중 오류 발생 ({uploaded_file.name}): {str(e)}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    progress_bar.empty()
    status_text.empty()
    return all_documents


# -------------------------------------------------------------
# Vector store creation and retrieval tool
# -------------------------------------------------------------
def create_vectorstore(uploaded_files):
    status_container = st.empty()

    try:
        # Step 1: Load
        status_container.info("🔄 **1단계**: 문서 파일을 읽고 있습니다...")
        all_documents = load_documents(uploaded_files)
        if not all_documents:
            status_container.error("❌ 문서를 읽을 수 없습니다.")
            return None

        # Step 2: Split
        status_container.info(f"🔄 **2단계**: 문서를 {len(all_documents)}개 항목으로 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_documents)

        # Step 3: Embed
        status_container.info(
            f"🔄 **3단계**: {len(split_docs)}개 청크 임베딩 중... (OpenAI API 호출)"
        )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = FAISS.from_documents(split_docs, embeddings)
        retriever = vector.as_retriever(search_kwargs={"k": 5})

        status_container.success(
            f"✅ **완료!** {len(uploaded_files)}개 파일, {len(split_docs)}개 청크 준비되었습니다!"
        )
        time.sleep(1)
        status_container.empty()

        def _format_location(md: dict) -> str:
            page = md.get("page", "")
            sheet = md.get("sheet", "")
            doc_type = md.get("type", "")
            if page != "":
                return f"{page}페이지"
            if sheet:
                return f"시트: {sheet}"
            if doc_type:
                return f"{doc_type} 문서"
            return ""

        def _retrieve(query: str):
            try:
                return retriever.get_relevant_documents(query)
            except Exception:
                return []

        def _format_docs_md(docs) -> str:
            if not docs:
                return "검색 결과가 없습니다."
            result_text = "### 검색된 문서 내용:\n\n"
            for idx, doc in enumerate(docs, 1):
                meta = getattr(doc, "metadata", {}) or {}
                source = meta.get("source", "알 수 없음")
                location = _format_location(meta)
                snippet = (doc.page_content or "").strip().replace("\n", " ")[:500]
                result_text += f"**[문서 {idx}]**\n"
                result_text += f"- 📄 파일명: {source}\n"
                if location:
                    result_text += f"- 📍 위치: {location}\n"
                result_text += f"- 📝 내용: {snippet}...\n\n"
                result_text += "---\n\n"
            return result_text

        # Tool callable (for Agent mode)
        def search_documents(query: str) -> str:
            docs = _retrieve(query)
            return _format_docs_md(docs)

        if Tool is None:
            retriever_tool = None
        else:
            retriever_tool = Tool(
                name="document_search",
                func=search_documents,
                description=(
                    "업로드된 문서(PDF, Excel, Word, PowerPoint, Text, CSV)에서 정보를 검색합니다. "
                    "검색 결과는 파일명, 페이지/시트 정보와 함께 반환됩니다. "
                    "반드시 이 도구를 사용하여 정보를 검색하세요."
                ),
            )

        # Provide everything we might need later
        return {
            "tool": retriever_tool,
            "retriever": retriever,
            "format_docs_md": _format_docs_md,
        }

    except Exception as e:
        status_container.error(f"❌ 벡터 DB 생성 중 오류: {str(e)}")
        return None


# =============================================================
# UI helpers
# =============================================================

def print_messages():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 


# =============================================================
# Main app
# =============================================================

def main():
    st.set_page_config(
        page_title="SCK 챗봇 Test 도우미",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded",
    )

    # Header
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if os.path.exists("./chatbot_logo.jpg"):
                st.image("./chatbot_logo.jpg", width=600)
            elif os.path.exists("chatbot_logo.jpg"):
                st.image("chatbot_logo.jpg", width=600)
        st.markdown("---")
        st.markdown("# SCK 챗봇 개발 도우미")
        st.markdown("### Document를 Upload하여 Test할 수 있습니다")
        st.markdown("**문의 사항은 기술팀 이현성 선임에게 진행해 주세요.**")
        st.markdown("**Contact:** 팀즈 or Mail ([hyunsung.lee@statschippac.com](mailto:hyunsung.lee@statschippac.com))")
        st.markdown("---")

    # Session state init
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state["vectorstore_ready"] = False
    if "agent_executor" not in st.session_state:
        st.session_state["agent_executor"] = None
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 설정")
        st.session_state["OPENAI_API"] = st.text_input(
            "OPENAI API 키",
            placeholder="sk-...",
            type="password",
            help="OpenAI API 키를 입력하세요",
        )

        st.markdown("---")
        st.header("📁 문서 업로드")
        with st.expander("📋 지원 파일 형식", expanded=False):
            st.markdown(
                """
                - 📕 PDF (.pdf)
                - 📊 Excel (.xlsx, .xls)
                - 📘 Word (.docx, .doc)
                - 🎨 PowerPoint (.pptx, .ppt)
                - 🌐 MHTML (.mhtml, .mht)
                - 📄 Text (.txt)
                - 📑 CSV (.csv)
                """
            )

        uploaded_files = st.file_uploader(
            "파일을 선택하세요",
            accept_multiple_files=True,
            type=[
                "pdf",
                "xlsx",
                "xls",
                "docx",
                "doc",
                "pptx",
                "ppt",
                "mhtml",
                "mht",
                "txt",
                "csv",
            ],
            key="file_uploader",
            help="여러 파일을 동시에 업로드할 수 있습니다",
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)}개의 파일 선택됨")
            with st.expander("📂 업로드된 파일 목록", expanded=True):
                total_size = 0
                for file in uploaded_files:
                    file_size = len(file.getvalue()) / 1024  # KB
                    total_size += file_size
                    file_icon = {
                        "pdf": "📕",
                        "xlsx": "📊",
                        "xls": "📊",
                        "docx": "📘",
                        "doc": "📘",
                        "pptx": "🎨",
                        "ppt": "🎨",
                        "mhtml": "🌐",
                        "mht": "🌐",
                        "txt": "📄",
                        "csv": "📑",
                    }.get(file.name.split(".")[-1].lower(), "📄")
                    st.markdown(f"{file_icon} **{file.name}**")
                    st.caption(f"크기: {file_size:.1f} KB")
                st.markdown(f"**총 용량**: {total_size:.1f} KB")

        if st.session_state["vectorstore_ready"]:
            st.markdown("---")
            st.header("📊 상태")
            st.metric("대화 수", len(st.session_state["messages"]) // 2)
            if st.button("🗑️ 대화 내역 초기화"):
                st.session_state["messages"] = []
                st.rerun()

    if not st.session_state["OPENAI_API"]:
        st.warning("⚠️ 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.info("💡 API 키는 기술팀 이현성 선임에게 문의 부탁드립니다.")
        return

    # Ensure OpenAI key in env before any embedding/LLM call
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API"]

    if not uploaded_files:
        st.info("📄 사이드바에서 문서 파일을 업로드해주세요.")
        st.markdown(
            """
            ### 💡 사용 방법
            1. 왼쪽 사이드바에서 OpenAI API 키 입력
            2. 분석하고 싶은 문서 파일 업로드 (여러 개 가능)
            3. 문서 처리 완료 후 질문 입력
            4. AI가 문서 내용을 기반으로 답변 제공

            ### ⚠️ 주의사항
            - 업로드된 문서는 OpenAI 서버로 전송됩니다
            - 기밀 문서는 업로드하지 마세요
            - 세션 종료 시 모든 데이터가 삭제됩니다
            """
        )
        return

    # Detect file changes to rebuild vectorstore
    current_file_names = [f.name for f in uploaded_files]
    if "previous_files" not in st.session_state:
        st.session_state["previous_files"] = []

    if current_file_names != st.session_state["previous_files"]:
        st.session_state["vectorstore_ready"] = False
        st.session_state["previous_files"] = current_file_names
        st.session_state["messages"] = []

    # Build vectorstore if needed
    if not st.session_state["vectorstore_ready"]:
        st.info("🔄 문서 업로드를 감지했습니다. 벡터 임베딩을 시작합니다...")
        vs = create_vectorstore(uploaded_files)
        if vs is None:
            st.error("❌ 문서 처리에 실패했습니다. 파일을 확인해주세요.")
            return

        # LLM init (support both model and model_name kwargs across versions)
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except TypeError:
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        st.session_state["llm"] = llm
        st.session_state["retriever"] = vs["retriever"]
        st.session_state["format_docs_md"] = vs["format_docs_md"]

        if _AGENTS_AVAILABLE and vs["tool"] is not None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "당신은 'SCK 챗봇 Test 도우미'입니다.\n\n"
                        "# 중요: 도구 사용 필수\n"
                        "질문을 받으면 반드시 먼저 document_search 도구를 호출하여 문서를 검색해야 합니다.\n"
                        "도구를 사용하지 않고 답변하면 안 됩니다.\n\n"
                        "# 답변 형식\n"
                        "## 요약\n핵심 내용 간단히 설명\n\n"
                        "## 상세 내용\n- 데이터는 표로 정리\n- 수치는 정확하게 (단위 포함)\n- 날짜/시간은 원본 형식 유지\n\n"
                        "## 참고 문서\n파일명과 위치 정보\n\n"
                        "# 규칙\n- 문서에 없는 내용은 추측 금지\n- 항상 한국어로 답변",
                    ),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            agent = create_tool_calling_agent(st.session_state["llm"], [vs["tool"]], prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=[vs["tool"]],
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                return_intermediate_steps=True,
            )
            st.session_state["agent_executor"] = agent_executor
        else:
            # Manual RAG fallback (no Agent): we'll compose our own prompt using retrieved context
            st.session_state["agent_executor"] = None

        # Celebration
        st.balloons()
        time.sleep(0.3)
        st.snow()
        st.success("🎉 문서 처리가 완료되었습니다! 이제 질문을 입력해주세요.")
        time.sleep(0.6)
        _debug_imports_banner()
        st.session_state["vectorstore_ready"] = True
        st.rerun()

    # Chat UI
    if st.session_state["vectorstore_ready"]:
        user_input = st.chat_input("💬 질문을 입력하세요...")
        if user_input and not st.session_state["is_processing"]:
            st.session_state["is_processing"] = True
            st.session_state["pending_question"] = user_input
            st.rerun()

        if st.session_state["is_processing"]:
            st.warning("⏳ 답변 생성 중입니다. 잠시만 기다려주세요...")

        print_messages()

        if st.session_state["pending_question"]:
            question = st.session_state["pending_question"]
            st.session_state["pending_question"] = None

            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.info("🔍 문서 검색 중...")

                try:
                    if st.session_state["agent_executor"] is not None:
                        # Agent path (will call the tool itself)
                        status_placeholder.info("🤖 답변 생성 중 (에이전트)...")
                        result = st.session_state["agent_executor"].invoke({"input": question})
                        response = result["output"]
                        status_placeholder.empty()
                        st.markdown(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                    else:
                        # Manual RAG path
                        status_placeholder.info("🤖 답변 생성 중 (수동 RAG)...")
                        retriever = st.session_state["retriever"]
                        docs = retriever.get_relevant_documents(question)
                        # Compose context text (compact)
                        context_snippets = []
                        for d in docs:
                            src = (d.metadata or {}).get("source", "")
                            snippet = (d.page_content or "").strip().replace("\n", " ")[:700]
                            context_snippets.append(f"[source: {src}] {snippet}")
                        context_text = "\n\n".join(context_snippets) if context_snippets else "(검색 결과 없음)"

                        manual_prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "당신은 업로드된 문서에서만 근거를 찾는 한국어 전문가 어시스턴트입니다.\n"
                                    "문서에 없는 내용은 추측하지 말고 '문서에 근거가 없습니다'라고 답하세요.\n"
                                    "요청된 포맷(요약/상세/참고 문서)을 반드시 지키세요.",
                                ),
                                (
                                    "human",
                                    "질문: {question}\n\n"
                                    "컨텍스트(검색 결과 발췌):\n{context}\n\n"
                                    "위 컨텍스트만 근거로 답하세요.",
                                ),
                            ]
                        )
                        try:
                            llm = st.session_state["llm"]
                        except KeyError:
                            try:
                                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                            except TypeError:
                                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

                        messages = manual_prompt.format_messages(question=question, context=context_text)
                        ai_msg = llm.invoke(messages)
                        response = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

                        # Add reference block (file and location list)
                        def _format_location(md: dict) -> str:
                            page = md.get("page", "")
                            sheet = md.get("sheet", "")
                            doc_type = md.get("type", "")
                            if page != "":
                                return f"{page}페이지"
                            if sheet:
                                return f"시트: {sheet}"
                            if doc_type:
                                return f"{doc_type} 문서"
                            return ""

                        refs_md_lines = ["\n\n## 참고 문서"]
                        if docs:
                            for i, d in enumerate(docs, 1):
                                md = d.metadata or {}
                                src = md.get("source", "알 수 없음")
                                loc = _format_location(md)
                                refs_md_lines.append(f"- [{i}] {src} " + (f"- {loc}" if loc else ""))
                        else:
                            refs_md_lines.append("- (검색 결과 없음)")

                        full_response = response + "\n" + "\n".join(refs_md_lines)

                        status_placeholder.empty()
                        st.markdown(full_response)
                        st.session_state["messages"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    status_placeholder.empty()
                    error_msg = f"❌ 오류가 발생했습니다:\n\n```{str(e)}```\n\n다시 시도해주세요."
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})

                finally:
                    st.session_state["is_processing"] = False
                    time.sleep(0.2)
                    st.rerun()


if __name__ == "__main__":
    main()
