import os
import streamlit as st
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredHTMLLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import tempfile
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
import pandas as pd
import time
from bs4 import BeautifulSoup
import email
from email import policy

# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 폭죽 효과 함수
def show_celebration():
    st.balloons()
    time.sleep(0.5)
    st.snow()

# 다양한 파일 형식 로드 함수
def load_documents(uploaded_files):
    all_documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📂 파일 처리 중: {uploaded_file.name} ({idx+1}/{total_files})")
        progress_bar.progress((idx + 1) / total_files)
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # 파일 형식별 로더 선택
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
            elif file_extension == 'txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source'] = uploaded_file.name
                
            elif file_extension == 'csv':
                df = pd.read_csv(tmp_file_path)
                text = df.to_string()
                documents = [Document(page_content=text, metadata={"source": uploaded_file.name, "type": "CSV"})]
                
            elif file_extension in ['xlsx', 'xls']:
                excel_file = pd.ExcelFile(tmp_file_path)
                documents = []
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(tmp_file_path, sheet_name=sheet_name)
                    text = f"Sheet: {sheet_name}\n\n{df.to_string()}"
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "source": uploaded_file.name, 
                            "sheet": sheet_name,
                            "type": "Excel"
                        }
                    ))
                    
            elif file_extension in ['docx', 'doc']:
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['type'] = 'Word'
                
            elif file_extension in ['pptx', 'ppt']:
                loader = UnstructuredPowerPointLoader(tmp_file_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['type'] = 'PowerPoint'
                
            elif file_extension in ['mhtml', 'mht']:
                # MHTML 파일 처리 (MIME 멀티파트)
                try:
                    with open(tmp_file_path, 'rb') as f:
                        msg = email.message_from_binary_file(f, policy=policy.default)
                    
                    # HTML 부분만 추출
                    text_parts = []
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type in ['text/html', 'text/plain']:
                            try:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    # 인코딩 감지 및 디코딩
                                    charset = part.get_content_charset() or 'utf-8'
                                    try:
                                        text = payload.decode(charset, errors='ignore')
                                    except:
                                        text = payload.decode('utf-8', errors='ignore')
                                    
                                    # HTML인 경우 BeautifulSoup으로 텍스트 추출
                                    if content_type == 'text/html':
                                        soup = BeautifulSoup(text, 'html.parser')
                                        text = soup.get_text(separator='\n', strip=True)
                                    
                                    text_parts.append(text)
                            except Exception as e:
                                continue
                    
                    if text_parts:
                        combined_text = '\n\n'.join(text_parts)
                        documents = [Document(
                            page_content=combined_text,
                            metadata={
                                "source": uploaded_file.name,
                                "type": "MHTML"
                            }
                        )]
                    else:
                        st.warning(f"⚠️ MHTML 파일에서 텍스트를 추출할 수 없습니다: {uploaded_file.name}")
                        continue
                        
                except Exception as e:
                    st.error(f"❌ MHTML 파일 처리 중 오류: {uploaded_file.name} - {str(e)}")
                    continue
                
            else:
                st.warning(f"⚠️ 지원하지 않는 파일 형식입니다: {uploaded_file.name}")
                continue
            
            all_documents.extend(documents)
            
        except Exception as e:
            st.error(f"❌ 파일 로드 중 오류 발생 ({uploaded_file.name}): {str(e)}")
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_documents

# 벡터DB 생성 (출처 정보 강제 포함)
def create_vectorstore(uploaded_files):
    status_container = st.empty()
    
    try:
        # 1단계: 문서 로드
        status_container.info("🔄 **1단계**: 문서 파일을 읽고 있습니다...")
        all_documents = load_documents(uploaded_files)
        
        if not all_documents:
            status_container.error("❌ 문서를 읽을 수 없습니다.")
            return None
        
        # 2단계: 텍스트 분할
        status_container.info(f"🔄 **2단계**: 문서를 {len(all_documents)}개의 섹션으로 분할하고 있습니다...")
        text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitters.split_documents(all_documents)
        
        # 3단계: 벡터 임베딩
        status_container.info(f"🔄 **3단계**: {len(split_docs)}개의 청크를 벡터로 변환하고 있습니다...\n(OpenAI API 호출 중 - 잠시만 기다려주세요)")
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        retriever = vector.as_retriever(search_kwargs={"k": 5})
        
        # 4단계: 완료
        status_container.success(f"✅ **완료!** {len(uploaded_files)}개 파일, {len(split_docs)}개 청크가 준비되었습니다!")
        time.sleep(1)
        status_container.empty()
        
        # Custom 검색 함수 - 출처 정보를 명시적으로 포함
        def search_documents(query: str) -> str:
            """문서 검색 및 출처 정보 포함"""
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "검색 결과가 없습니다."
            
            result_text = "### 검색된 문서 내용:\n\n"
            
            for idx, doc in enumerate(docs, 1):
                # 메타데이터 추출
                source = doc.metadata.get('source', '알 수 없음')
                page = doc.metadata.get('page', '')
                sheet = doc.metadata.get('sheet', '')
                doc_type = doc.metadata.get('type', '')
                
                # 출처 정보 구성
                location = ""
                if page != '':
                    location = f"{page}페이지"
                elif sheet:
                    location = f"시트: {sheet}"
                elif doc_type:
                    location = f"{doc_type} 문서"
                
                # 결과 포맷팅
                result_text += f"**[문서 {idx}]**\n"
                result_text += f"- 📄 파일명: {source}\n"
                if location:
                    result_text += f"- 📍 위치: {location}\n"
                result_text += f"- 📝 내용:\n{doc.page_content[:500]}...\n\n"
                result_text += "---\n\n"
            
            return result_text
        
        # Tool로 변환
        from langchain.agents import Tool
        
        retriever_tool = Tool(
            name="document_search",
            func=search_documents,
            description=(
                "업로드된 문서(PDF, Excel, Word, PowerPoint, Text, CSV)에서 정보를 검색합니다. "
                "검색 결과는 파일명, 페이지/시트 정보와 함께 반환됩니다. "
                "반드시 이 도구를 사용하여 정보를 검색하세요."
            )
        )
        
        return retriever_tool
        
    except Exception as e:
        status_container.error(f"❌ 벡터 DB 생성 중 오류: {str(e)}")
        return None

# 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# 메인 실행
def main():
    st.set_page_config(
        page_title="SCK 챗봇 Test 도우미", 
        layout="wide", 
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )

    # 헤더 섹션
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if os.path.exists('./chatbot_logo.jpg'):
                st.image('./chatbot_logo.jpg', width=600)
            elif os.path.exists('chatbot_logo.jpg'):
                st.image('chatbot_logo.jpg', width=600)
        
        st.markdown('---')
        st.markdown("# SCK 챗봇 개발 도우미")
        st.markdown("### Document를 Upload하여 Test할 수 있습니다")
        st.markdown("**문의 사항은 기술팀 이현성 선임에게 진행해 주세요.**")
        st.markdown("**Contact:** 팀즈 or Mail ([hyunsung.lee@statschippac.com](mailto:hyunsung.lee@statschippac.com))")
        st.markdown('---')

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}
    if "vectorstore_ready" not in st.session_state:
        st.session_state["vectorstore_ready"] = False
    if "agent_executor" not in st.session_state:
        st.session_state["agent_executor"] = None
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        st.session_state["OPENAI_API"] = st.text_input(
            "OPENAI API 키", 
            placeholder="sk-...", 
            type="password",
            help="OpenAI API 키를 입력하세요"
        )
        
        st.markdown('---')
        st.header("📁 문서 업로드")
        
        # 지원 파일 형식 안내
        with st.expander("📋 지원 파일 형식", expanded=False):
            st.markdown("""
            - 📕 PDF (.pdf)
            - 📊 Excel (.xlsx, .xls)
            - 📘 Word (.docx, .doc)
            - 🎨 PowerPoint (.pptx, .ppt)
            - 🌐 MHTML (.mhtml, .mht)
            - 📄 Text (.txt)
            - 📑 CSV (.csv)
            """)
        
        uploaded_files = st.file_uploader(
            "파일을 선택하세요", 
            accept_multiple_files=True, 
            type=['pdf', 'xlsx', 'xls', 'docx', 'doc', 'pptx', 'ppt', 'mhtml', 'mht', 'txt', 'csv'],
            key="file_uploader",
            help="여러 파일을 동시에 업로드할 수 있습니다"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)}개의 파일 선택됨")
            
            # 업로드된 파일 목록 표시
            with st.expander("📂 업로드된 파일 목록", expanded=True):
                total_size = 0
                for file in uploaded_files:
                    file_size = len(file.getvalue()) / 1024  # KB
                    total_size += file_size
                    file_icon = {
                        'pdf': '📕', 'xlsx': '📊', 'xls': '📊',
                        'docx': '📘', 'doc': '📘', 'pptx': '🎨',
                        'ppt': '🎨', 'mhtml': '🌐', 'mht': '🌐',
                        'txt': '📄', 'csv': '📑'
                    }.get(file.name.split('.')[-1].lower(), '📄')
                    
                    st.markdown(f"{file_icon} **{file.name}**")
                    st.caption(f"크기: {file_size:.1f} KB")
                
                st.markdown(f"**총 용량**: {total_size:.1f} KB")
        
        # 통계 정보
        if st.session_state["vectorstore_ready"]:
            st.markdown('---')
            st.header("📊 세션 정보")
            st.metric("대화 수", len(st.session_state["messages"]) // 2)
            if st.button("🗑️ 대화 내역 초기화"):
                st.session_state["messages"] = []
                st.session_state["session_history"] = {}
                st.rerun()

    # 메인 영역
    if not st.session_state["OPENAI_API"]:
        st.warning("⚠️ 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.info("💡 API 키는 기술팀 이현성 선임에게 문의 부탁드립니다.")
        return
    
    if not uploaded_files:
        st.info("📄 사이드바에서 문서 파일을 업로드해주세요.")
        st.markdown("""
        ### 💡 사용 방법
        1. 왼쪽 사이드바에서 OpenAI API 키 입력
        2. 분석하고 싶은 문서 파일 업로드 (여러 개 가능)
        3. 문서 처리 완료 후 질문 입력
        4. AI가 문서 내용을 기반으로 답변 제공
        
        ### ⚠️ 주의사항
        - 업로드된 문서는 OpenAI 서버로 전송됩니다
        - 기밀 문서는 업로드하지 마세요
        - 세션 종료 시 모든 데이터가 삭제됩니다
        """)
        return

    # 벡터스토어 생성 (파일이 변경되었을 때만)
    os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
    
    current_file_names = [f.name for f in uploaded_files]
    
    if "previous_files" not in st.session_state:
        st.session_state["previous_files"] = []
    
    # 파일이 변경되었는지 확인
    if current_file_names != st.session_state["previous_files"]:
        st.session_state["vectorstore_ready"] = False
        st.session_state["previous_files"] = current_file_names
        st.session_state["messages"] = []
        st.session_state["session_history"] = {}
    
    if not st.session_state["vectorstore_ready"]:
        with st.container():
            st.info("🔄 문서 업로드를 감지했습니다. 벡터 임베딩을 시작합니다...")
            
            doc_search = create_vectorstore(uploaded_files)
            
            if doc_search is None:
                st.error("❌ 문서 처리에 실패했습니다. 파일을 확인해주세요.")
                return
            
            # LLM 및 Agent 설정
            tools = [doc_search]
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                "당신은 'SCK 챗봇 Test 도우미'입니다.\n\n"
                
                "# 중요: 도구 사용 필수\n"
                "질문을 받으면 반드시 먼저 document_search 도구를 호출하여 문서를 검색해야 합니다.\n"
                "도구를 사용하지 않고 답변하면 안 됩니다.\n\n"
                
                "# 작동 순서\n"
                "1. 질문 받음\n"
                "2. document_search 도구 실행 (필수)\n"
                "3. 검색 결과를 바탕으로 답변 작성\n"
                "4. 출처 정보 포함\n\n"
                
                "# 답변 형식\n"
                "## 요약\n"
                "핵심 내용 간단히 설명\n\n"
                
                "## 상세 내용\n"
                "- 데이터는 표로 정리\n"
                "- 수치는 정확하게 (단위 포함)\n"
                "- 날짜/시간은 원본 형식 유지\n\n"
                
                "## 참고 문서\n"
                "파일명과 위치 정보\n\n"
                
                "# 규칙\n"
                "- 문서에 없는 내용은 추측 금지\n"
                "- 항상 한국어로 답변\n"
                "- 반드시 도구를 먼저 사용"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15,
                early_stopping_method="force",
                return_intermediate_steps=True
            )
            
            st.session_state["agent_executor"] = agent_executor
            st.session_state["vectorstore_ready"] = True
            
            # 폭죽 효과
            show_celebration()
            st.success("🎉 문서 처리가 완료되었습니다! 이제 질문을 입력해주세요.")
            time.sleep(1)
            st.rerun()

    # 채팅 인터페이스
    if st.session_state["vectorstore_ready"]:
        # 질문 입력창 - 최상단에 배치하여 우선 렌더링
        user_input = st.chat_input(
            '💬 질문을 입력하세요...',
            disabled=st.session_state["is_processing"],
            key="chat_input"
        )
        
        # 새 질문 입력 시 - 즉시 처리
        if user_input and not st.session_state["is_processing"]:
            st.session_state["is_processing"] = True
            st.session_state["pending_question"] = user_input
            st.rerun()
        
        # 처리 중 상태 표시
        if st.session_state["is_processing"]:
            st.warning("⏳ 답변 생성 중입니다. 잠시만 기다려주세요...")
        
        # 이전 메시지 출력
        print_messages()
        
        # 대기 중인 질문 처리
        if st.session_state["pending_question"]:
            question = st.session_state["pending_question"]
            st.session_state["pending_question"] = None
            
            # 사용자 메시지 추가 및 표시
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.info("🔍 문서 검색 중...")
                
                try:
                    session_id = "default_session"
                    session_history = get_session_history(session_id)
                    
                    # Agent 실행
                    status_placeholder.info("🤖 답변 생성 중...")
                    result = st.session_state["agent_executor"].invoke({"input": question})
                    response = result['output']
                    
                    status_placeholder.empty()
                    st.markdown(response)
                    
                    # 메시지 저장
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    session_history.add_message({"role": "user", "content": question})
                    session_history.add_message({"role": "assistant", "content": response})
                    
                except Exception as e:
                    status_placeholder.empty()
                    error_msg = f"❌ 오류가 발생했습니다:\n\n```\n{str(e)}\n```\n\n다시 시도해주세요."
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                
                finally:
                    # 처리 완료
                    st.session_state["is_processing"] = False
                    time.sleep(0.3)  # 짧은 딜레이 후 리렌더링
                    st.rerun()

if __name__ == "__main__":
    main()