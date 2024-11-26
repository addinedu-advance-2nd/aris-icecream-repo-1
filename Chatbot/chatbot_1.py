import sys
import os
import warnings
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
import nest_asyncio

# RAG Chatbot Imports
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import re
import ast
from langgraph.graph import END, StateGraph, START
#from funtions import retrieve, generate, grade_documents, transform_query, web_search, route_question, decide_to_generate, grade_generation_v_documents_and_question
from funtions import retrieve, generate, grade_documents, transform_query, web_search,  decide_to_generate, grade_generation_v_documents_and_question
from langchain_core.prompts import ChatPromptTemplate

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QTextEdit, QLineEdit, \
    QPushButton, QLabel

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLineEdit, QLabel, QComboBox

import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit

nest_asyncio.apply()
warnings.filterwarnings('ignore')
import speech_recognition as sr

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain.document_loaders import YoutubeLoader

memory = MemorySaver()

global_text_edit = None  # 초기화 (전역 변수)

# 메모리 세팅 시 필요한 키들
memory_config = {
    "thread_id": "your_thread_id",
    "checkpoint_ns": "your_namespace",
    "checkpoint_id": "your_checkpoint_id"
}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    focus: Literal["web", "academic", "video", "math"]
    memory: dict  # 메모리를 추가합니다.

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
# Define tools and chatbot function
web_tool = TavilySearchResults(max_results=2)

#print(fr'web_tool : {web_tool}')

@tool
def academic_tool(query: str):
    """Academic paper search tool"""
    arxiv = ArxivAPIWrapper()
    docs = arxiv.run(query)
    return docs

@tool
def math_tool(query: str):
    """Math tool"""
    wolfram = WolframAlphaAPIWrapper()
    result = wolfram.run(query)
    return result

youtube_search_tool = YouTubeSearchTool()


@tool
def video_tool(query:str) -> str:
    """
    Retriever tool for the transcript of a YouTube video.
    If user want to find some information, this tool is good to gather youtube video information.
    query should be given in string format.
    """
    #query에 해당하는 Youtube 비디오 URL 가져오기
    urls = youtube_search_tool.run(query)
    urls = ast.literal_eval(urls)

    urls = ["https://www.youtube.com/shorts/yQPzlXyJcNg"]

    #URL 순회하면서 Document 객체에 내용 담기
    docs = []
    for url in urls:
        loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
        language=["en", "ko"]
        )
        scripts = loader.load()
        script_content = scripts[0].page_content
        title=scripts[0].metadata['title']
        author=scripts[0].metadata['author']
        doc = Document(page_content=script_content, metadata={"source": url, "title":title, "author":author})
        docs.append(doc)

    #모든 비디오의 내용을 벡터DB에 담기
    text_splitter = RecursiveCharacterTextSplitter(
        separators  = ["\n\n", "\n", ".", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    retrieved_docs = retriever.invoke(query)

    video_results = []

    for doc in retrieved_docs:
        title = doc.metadata.get('title', 'No title available')
        author = doc.metadata.get('author', 'No author available')
        script_content = doc.page_content

        video_info = f"""
        Video Information:
        ------------------
        Title: {title}
        Author: {author}
        Transcript:
        {script_content}
        ------------------
        """
        video_results.append(video_info)

    # Join all video results into a single string
    all_video_results = "\n\n".join(video_results)

    return all_video_results

tools = {
    "web": [web_tool],
    "academic": [academic_tool],
    "video": [video_tool],
    "math": [math_tool]
}
tool_nodes = {focus: ToolNode(tools[focus]) for focus in tools}

print(fr'tool_nodes 설정 :{tool_nodes}')

# Define chatbot function
llm = ChatOpenAI(model="gpt-4o-mini")


def filter_messages(messages: list):
    # This is very simple helper function which only ever uses the last message
    return messages[-2:]


# def chatbot(state: State):
#     llm_with_tools = llm.bind_tools(tools[state["focus"]])
#     messages = filter_messages(state["messages"])
#     result = llm_with_tools.invoke(messages)

def chatbot(state: State):
    llm_with_tools = llm.bind_tools(tools[state["focus"]])

    messages = filter_messages(state["messages"])
    result = llm_with_tools.invoke(messages)

    return {"messages": [result]}

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    QApplication.processEvents()

    current_text = global_user_message

    question = current_text

    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
            The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
            Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    system = """
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search.
    If the user tells you their name, remember it. When the user asks "What's my name?" or similar questions, reply with the name they provided.
    """
    system = """
        You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search.
        If the user tells you their name, remember it. When the user asks "What's my name?" or similar questions, reply with the name they provided.
        If the user asks "What's my name?" or a similar question after telling you their name, do not perform a web search. Simply provide the name they shared.
    """

    system = """당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
                    벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
                    이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오.
                    문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오."""
    # system = """
    # 당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
    # 벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
    # 이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오.
    # 문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오.
    #
    # 사용자가 주문할 수 있는 메뉴는 다음과 같습니다:
    # - 바나나맛
    # - 초코맛
    # - 딸기맛
    #
    # 토핑 옵션:
    # - 로투스
    # - 레인보우
    # - 오레오
    #
    # 사용자가 선택한 메뉴와 토핑, 개수에 따라 맞춤형 답변을 제공합니다.
    # 만약 사용자가 주문을 하면, 주문 내역을 확인하고 맞는지 확인합니다. 그 외의 질문에는 일반적인 응답을 제공합니다.
    # """

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        #self.text_edit.append("---ROUTE QUESTION TO WEB SEARCH---")
        QApplication.processEvents()
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        #self.text_edit.append("---ROUTE QUESTION TO RAG---")
        QApplication.processEvents()
        return "vectorstore"

# Build graph
workflow = StateGraph(State)
#workflow.add_node("chatbot", chatbot)

for focus, tool_node in tool_nodes.items():
    workflow.add_node(f"{focus}_tools", tool_node)

#workflow.add_node("web_search", web_search)  # web search
workflow.add_node("web_search", chatbot)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
#workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
#workflow.add_node("chatbot", chatbot)  # chatbot

def tools_condition(state):

    if state["messages"][-1].tool_calls:

        return f"{state['focus']}_tools"
    return END

for focus in tools:
    workflow.add_edge(f"{focus}_tools", "web_search")

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },

        )
# workflow.add_edge("web_search", "generate")web_search
#workflow.add_edge("web_search", "chatbot")
workflow.add_conditional_edges(
    "web_search",
    tools_condition,
    {
        "web_tools": "web_tools",
        "academic_tools": "academic_tools",
        "video_tools": "video_tools",
        "math_tools": "math_tools",
        END: END
    }
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
         "transform_query": "transform_query",
        "generate": "generate",
     },
 )
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
           "not supported": "generate",
          "useful": END,
           "not useful": "transform_query",
    },
 )

#graph = workflow.compile()
graph = workflow.compile(checkpointer=memory_config)


global_user_message = ""
global_chat_display = ""

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()

        # 상태 관리용 변수
        self.messages = []
        self.memory = {}  # 메모리를 저장할 딕셔너리 추가

        # UI 구성
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Chat App')

        layout = QVBoxLayout()

        # 메시지 표시용 텍스트 창
        self.chat_display = QTextEdit(self)

        global global_text_edit  # 전역 변수를 사용한다고 명시
        self.text_edit = self.chat_display  # QTextEdit 위젯을 생성
        global_text_edit = self.text_edit  # 전역 변수에 QTextEdit 객체를 저장

        self.chat_display.setReadOnly(True)  # 메시지 입력 불가
        layout.addWidget(self.chat_display)

        # 메시지 입력 필드
        self.message_input = QLineEdit(self)

        #self.text_edit.setPlaceholderText("여기에 음성 질문이 표시됩니다...")
        self.message_input.setPlaceholderText("여기에 음성 질문이 표시됩니다.")

        global_text_edit = self.message_input
        layout.addWidget(self.message_input)


        # 체크박스와 아이콘을 추가하는 방법
        self.checkbox_web = QCheckBox("web", self)
        self.checkbox_web.setIcon(QIcon("web-link.png"))  # 아이콘 설정
        self.checkbox_web.setIconSize(QSize(30, 30))  # 아이콘 크기 설정
        layout.addWidget(self.checkbox_web)

        self.checkbox_academic = QCheckBox("academic", self)
        self.checkbox_academic.setIcon(QIcon("knowledge.png"))
        self.checkbox_academic.setIconSize(QSize(30, 30))
        layout.addWidget(self.checkbox_academic)

        self.checkbox_video = QCheckBox("video", self)
        self.checkbox_video.setIcon(QIcon("find.png"))
        self.checkbox_video.setIconSize(QSize(30, 30))
        layout.addWidget(self.checkbox_video)

        self.checkbox_math = QCheckBox("math", self)
        self.checkbox_math.setIcon(QIcon("search.png"))
        self.checkbox_math.setIconSize(QSize(30, 30))
        layout.addWidget(self.checkbox_math)

        # 전송 버튼
        # send_button = QPushButton('Send', self)
        send_button = QPushButton('음성 인식 시작', self)
        self.send_button = send_button

        send_button.clicked.connect(self.start_voice_recognition)

        layout.addWidget(send_button)
        self.send_button.setStyleSheet("background-color: #007BFF; color: white; font-size: 16px;")

        # 레이아웃 설정
        self.setLayout(layout)
        self.resize(400, 300)
        self.resize(800, 1000)


        # QTextEdit 스타일 (파란색 배경과 흰색 텍스트)
        self.chat_display.setStyleSheet("background-color: lightblue; color: darkblue; font-size: 16px;")

        # 스타일시트 설정 (파란색 계통으로)
        self.setStyleSheet("""
                    QWidget {
                        background-color: #f0f8ff;  # 아주 연한 파란색 배경
                        color: #00008b;  # 어두운 파란색 글자 색
                    }

                    QTextEdit {
                        background-color: #e0ffff;  # 연한 파란색 배경
                        border: 1px solid #87cefa;  # 밝은 파란색 테두리
                        color: #00008b;  # 어두운 파란색 글자 색
                    }

                    QLineEdit {
                        background-color: #e0ffff;
                        border: 1px solid #87cefa;
                        color: #00008b;
                    }

                    QPushButton {
                        background-color: #4682b4;  # 파란색 배경
                        color: white;  # 흰색 글자
                        border: none;
                        padding: 10px;
                        border-radius: 5px;
                    }

                    QPushButton:hover {
                        background-color: #5f9ea0;  # 버튼 호버 시 밝은 파란색
                    }

                    QCheckBox {
                        color: #00008b;  # 체크박스 글자 색
                        font-weight: bold;
                    }

                    QCheckBox::indicator {
                        border: 1px solid #87cefa;  # 체크박스 테두리 색
                        background-color: #f0f8ff;
                        width: 20px;
                        height: 20px;
                    }

                    QCheckBox::indicator:checked {
                        background-color: #4682b4;  # 체크된 체크박스 배경 색
                        border-color: #4682b4;
                    }

                    QCheckBox::indicator:unchecked {
                        background-color: #f0f8ff;
                        border-color: #87cefa;
                    }
                """)

    # 메뉴, 토핑, 개수 변수 정의
    menu = ["바나나맛", "초코맛", "딸기맛"]
    toppings = ["로투스", "레인보우", "오레오"]
    quantity = 2  # 예시로 2개를 선택한 경우

    def store_order(self, user_id, order_list):

        # 사용자 별로 주문 정보를 저장
        if user_id not in self.memory:
            self.memory[user_id] = {'order_history': []}

        # 주문 정보를 order_history에 추가

        self.memory[user_id]['order_history'].append(order_list)

    def show_order_confirmation(self, order_list):

        for order in order_list:

        confirmation = "예"

        if confirmation == "예":
            print("주문이 완료되었습니다. 감사합니다!")
        else:
            print("주문이 취소되었습니다. 다시 선택해 주세요.")

        self.send_message2(order_list)


    def start_voice_recognition(self):


            recognizer = sr.Recognizer()
            with sr.Microphone() as source:  # 마이크 열기

                recognizer.adjust_for_ambient_noise(source)

                # 마이크 입력 받기
                audio = recognizer.listen(source)

            try:
                mySpeech = recognizer.recognize_google(audio, language='ko-KR', show_all=False)

                # self.text_edit.setText(mySpeech)  # 텍스트 박스에 음성 결과 표시
                self.message_input.setText(mySpeech)  # 텍스트 박스에 음성 결과 표시

                import re

                # 주문 내역 예시
                order_text = "초코맛 아이스크림 1개, 레인보우 토핑 추가랑 딸기맛 아이스크림 2개, 오레오 토핑 추가"

                pattern = r"(\w+맛)\s(아이스크림)\s(\d+)개,\s(\w+)\s토핑\s추가"

                # 정규식 매칭
                matches = re.findall(pattern, order_text)

                # 결과 출력
                order_list = []

                if matches:
                    for match in matches:

                        order = {
                            "메뉴": match[0],
                            # "종류": match[1],
                            "수량": int(match[2]),
                            "토핑": match[3]
                        }

                        order_list.append(order)
                    # 주문 목록 출력
                    print(fr'order_list : {order_list}')

                    self.show_order_confirmation(order_list)
                else:
                    print("주문 관련 내용이 아닙니다.")

                self.send_message()


            except sr.UnknownValueError:
                print("Google 음성 인식이 오디오를 이해할 수 없습니다.")

                mySpeech = "아이스크림이 안 차가워"

                self.message_input.setText(mySpeech)  # 텍스트 박스에 음성 결과 표시

                import re

                # 주문 내역 예시
                order_text = "초코맛 아이스크림 1개, 레인보우 토핑 추가랑 딸기맛 아이스크림 2개, 오레오 토핑 추가"

                pattern = r"(\w+맛)\s(아이스크림)\s(\d+)개,\s(\w+)\s토핑\s추가"

                # 정규식 매칭
                matches = re.findall(pattern, order_text)

                # 결과 출력
                order_list = []

                if matches:
                    for match in matches:

                        order = {
                            "메뉴": match[0],
                            # "종류": match[1],
                            "수량": int(match[2]),
                            "토핑": match[3]
                        }

                        order_list.append(order)
                    # 주문 목록 출력
                    print(fr'order_list : {order_list}')

                    self.show_order_confirmation(order_list)
                else:
                    print("주문 관련 내용이 아닙니다.")



            except sr.RequestError as e:
                print("Google 음성 인식 서비스에서 결과를 요청할 수 없습니다.; {0}".format(e))

                from transformers import pipeline




    def create_checkbox(self, layout, label, icon_path):

        checkbox = QCheckBox(label)
        checkbox.setIcon(QIcon(icon_path))  # Set the icon for the checkbox
        checkbox.setIconSize(QSize(30, 30))  # Set the icon size using QSize
        layout.addWidget(checkbox)

        return checkbox

    def get_selected_focus(self):


        # 체크박스에서 체크된 항목 가져오기
        selected_focus = []

        if self.checkbox_web.isChecked():
            selected_focus.append("web")
        if self.checkbox_academic.isChecked():
            selected_focus.append("academic")
        if self.checkbox_video.isChecked():
            selected_focus.append("video")
        if self.checkbox_math.isChecked():
            selected_focus.append("math")

        return selected_focus

    def route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        self.text_edit.append("---ROUTE QUESTION---")
        QApplication.processEvents()
        question = state["question"]


        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
                The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        #         system = """당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
        # 벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
        # 이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오."""
        #         system = """당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
        #         벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
        #         이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오.
        #         문서는 영어로 되어 있을 수 있지만, 답변은 한글로 번역하여 제공하십시오."""
        #         system = """당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
        #                 벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
        #                 이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오.
        #                 문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오."""

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        # 텍스트 박스의 현재 내용을 가져와 변수에 저장
        current_text = self.text_edit.toPlainText()  # 현재 텍스트 가져오기

        global global_user_message  # 전역 변수를 사용한다고 명시
        user_message = self.message_input.text()  # 메시지 입력란에서 텍스트 가져오기
        global_user_message = user_message  # 전역 변수에 저장

        print(f'현재 텍스트: {current_text}')  # 현재 텍스트 출력
        print(f'global_user_message: {global_user_message}')  # 현재 텍스트 출력

        question_router = route_prompt | structured_llm_router

        print(
            question_router.invoke(

                {"question": current_text}

            )
        )


        source = question_router.invoke({"question": question})

        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            self.text_edit.append("---ROUTE QUESTION TO WEB SEARCH---")
            QApplication.processEvents()
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            self.text_edit.append("---ROUTE QUESTION TO RAG---")
            QApplication.processEvents()
            return "vectorstore"

    def send_message2(self, order_list):

            user_id = '001'

            # 입력 메시지 가져오기
            user_message = self.message_input.text()

            global glostore_orderuser_message  # 전역 변수를 사용한다고 명시
            user_message = self.message_input.text()  # 메시지 입력란에서 텍스트 가져오기
            user_message = "초코맛 아이스크림 1개, 레인보우 토핑 추가랑 딸기맛 아이스크림 2개, 오레오 토핑 추가"
            global_user_message = user_message  # 전역 변수에 저장



            if user_message:

                selected_focus = self.get_selected_focus()  # self를 사용하여 get_selected_focus 호출

                # self.messages.append({"role": "assistant", "focus": selected_focus, "content": assistant_response})
                self.messages.append({"role": "user", "focus": selected_focus, "content": user_message})

                self.update_chat_display()

                graph = workflow.compile(checkpointer=memory)
                config2 = {
                    "configurable": {
                        "thread_id": "2",  # thread_id 값 설정
                        "memory": self.memory  # 메모리 기능을 추가
                    }
                }

                responses = []

                # 먼저 메모리 설정을 추가한 config2 객체 정의
                config2 = {
                    "configurable": {
                        "thread_id": "2",  # thread_id 값 설정
                        "memory": self.memory  # 메모리 기능을 추가
                    }
                }


                # 선택된 focus가 여러 개라면 각각에 대해 invoke 실행

                selected_focus_tuple = tuple(selected_focus)
                selected_focus = selected_focus_tuple
                selected_focus_tuple = ('주문')
                selected_focus = selected_focus_tuple

                selected_focus = '.'
                for focus in selected_focus:

                    user_id = '001'


                    # 주문 내역을 저장할 변수
                    order_confirmation_message = "주문 내역을 확인합니다:\n"


                    print(f"주문 내역을 확인합니다:")
                    for order in order_list:
                        print(f"메뉴: {order['메뉴']}, 수량: {order['수량']}, 토핑: {order['토핑']}")
                        order_confirmation_message += f"메뉴: {order['메뉴']}, 수량: {order['수량']}, 토핑: {order['토핑']}\n"

                    confirmation = "예"
                    # confirmation = input("이 내용으로 주문을 확정하시겠습니까? (예/아니오): ")

                    if confirmation == "예":
                        print("주문이 완료되었습니다. 감사합니다!")
                        order_confirmation_message += "주문이 완료되었습니다. 감사합니다!"
                    else:
                        print("주문이 취소되었습니다. 다시 선택해 주세요.")
                        order_confirmation_message += "주문이 취소되었습니다. 다시 선택해 주세요."

                    result = order_confirmation_message

                    #self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message]

                    # 메모리에 result 저장 (선택된 focus에 해당하는 값으로 result 저장)
                    self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message, result]


                    # 응답 처리
                    if "messages" in result and len(result["messages"]) > 0:
                        assistant_response = result["messages"][-1].content

                        print(f'assistant_response : {assistant_response}')
                        responses.append((selected_focus, assistant_response))
                    else:
                        assistant_response = result
                        responses.append((selected_focus, assistant_response))

                    self.messages.append({"role": "assistant", "focus": focus, "content": assistant_response})

                    responses = []

                    # UI 업데이트
                    self.update_chat_display()

            # 입력 필드 초기화
            self.message_input.clear()

    def send_message(self):
        # 입력 메시지 가져오기
        user_message = self.message_input.text()

        global global_user_message  # 전역 변수를 사용한다고 명시
        user_message = self.message_input.text()  # 메시지 입력란에서 텍스트 가져오기
        global_user_message = user_message  # 전역 변수에 저장

        if user_message:

            selected_focus = self.get_selected_focus()  # self를 사용하여 get_selected_focus 호출

            self.messages.append({"role": "user", "focus": selected_focus, "content": user_message})
            print(f'self.messages: {self.messages}')

            self.update_chat_display()

            graph = workflow.compile(checkpointer=memory)
            config2 = {
                "configurable": {
                    "thread_id": "2",  # thread_id 값 설정
                    "memory": self.memory  # 메모리 기능을 추가
                }
            }

            responses = []

            config2 = {
                "configurable": {
                    "thread_id": "2",  # thread_id 값 설정
                    "memory": self.memory  # 메모리 기능을 추가
                }
            }

            print(fr'self.memory  : {self.memory }')

            # 선택된 focus가 여러 개라면 각각에 대해 invoke 실행
            print(fr'selected_focus : {selected_focus}')
            selected_focus_tuple = tuple(selected_focus)
            selected_focus = selected_focus_tuple

            for focus in selected_focus:
                print(fr' send_message focus : {focus}')

                system_message = """
                                    당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
                                    벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
                                    이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그 외의 경우에는 웹 검색을 기본 설정으로 사용하십시오.
                                    문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오.

                                    사용자가 주문할 수 있는 메뉴는 다음과 같습니다:
                                    - 바나나맛
                                    - 초코맛
                                    - 딸기맛

                                    토핑 옵션:
                                    - 로투스
                                    - 레인보우
                                    - 오레오

                                    ### 검색 및 주문 접수 모드
                                    - 사용자가 검색 옵션을 선택하지 않으면, 기본적으로 웹 검색을 설정하여 검색합니다.
                                    - 사용자가 주문을 의도하는 발화를 할 경우, 검색 옵션 설정 없이 주문 접수 모드로 전환하여 주문을 받습니다.

                                    ### 주문 접수 모드
                                    기본적으로 AI 에이전시처럼 사용자 질문에 일반적인 응답을 제공하되, 사용자가 주문을 의도하는 발화를 할 경우 자동으로 주문 접수 모드로 전환하여 대응합니다.

                                    1. 사용자가 아이스크림 맛을 이미 언급한 경우(바나나맛, 초코맛, 딸기맛 중 하나), 해당 맛에 대한 질문은 생략하고 진행합니다.
                                    2. 사용자가 토핑을 이미 언급한 경우(로투스, 레인보우, 오레오 중 하나), 토핑에 대한 질문은 생략하고 진행합니다.
                                    3. 사용자가 개수를 이미 언급한 경우, 추가로 개수를 묻지 않고 바로 최종 주문 확인으로 넘어갑니다.

                                    이전 대화에서 사용자가 선택한 아이스크림 맛, 토핑, 개수를 기억하여 중복된 질문을 피하도록 설정합니다.

                                    **주문 진행 메시지 예시**:
                                    - 바나나맛 아이스크림을 주문하셨습니다. 추가할 토핑은 무엇으로 하시겠습니까? (로투스, 레인보우, 오레오 중에서 선택해 주세요.)
                                    - 오레오 토핑을 선택하셨습니다. 주문 개수는 몇 개로 하시겠습니까?
                                    - 주문하신 바나나맛 아이스크림에 오레오 토핑을 추가하여 1개 주문을 확인합니다. 주문을 완료하시겠습니까?

                                    **최종 주문 확인 메시지**:
                                    - 사용자가 긍정적으로 응답할 경우 (예: "응", "네", "그래", "좋아", "예", "맞아", "확인") 주문이 완료됩니다.
                                    - 주문이 완료되었습니다. 감사합니다!

                                    주문 접수 이외의 질문에는 AI 에이전시로서 사용자 질문에 적절히 대응하십시오.
                                    """

                global m

                # 대화 히스토리를 불러오고, 메모리에 이전 응답 저장
                m = self.memory.get(selected_focus, [])
                print(fr'm :{m}')
                # 사용자 메시지와 포커스를 기반으로 프롬프트 생성
                prompt = system_message + f"\n사용자 메시지: {user_message}\n현재 포커스: {focus}\n대화 히스토리: {m}"

                # 호출
                result = graph.invoke({
                    "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                    "focus": focus,
                }, config2)  # config2는 필요한 설정을 포함

                print(fr'result : {result}')

                print(fr'result: {result}')
                self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message]

                m += self.memory.get(selected_focus, []) + [user_message, result]

                #self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message]

                # 메모리에 result 저장 (선택된 focus에 해당하는 값으로 result 저장)
                self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message, result]

                # 응답 처리
                if "messages" in result and len(result["messages"]) > 0:
                    assistant_response = result["messages"][-1].content

                    responses.append((selected_focus, assistant_response))

                self.messages.append({"role": "assistant", "focus": focus, "content": assistant_response})
                print(f'self.messages: {self.messages}')

                responses = []


                if "messages" in result and len(result["messages"]) > 0:
                    assistant_response = result["messages"][-1].content

                    print(f'assistant_response: {assistant_response}')

                    assistant_response = {
                        "role": "assistant",
                        "focus": focus,  # focus 정보를 추가
                        "content": result["messages"][-1].content  # 응답 내용 추가
                    }


                    responses.append((selected_focus, assistant_response))
                    print(f'responses: {responses}')

                # 메모리에 봇의 응답도 저장
                self.memory[selected_focus].append(assistant_response)

                # UI 업데이트
                self.update_chat_display()


        # 입력 필드 초기화
        self.message_input.clear()

    def update_chat_display(self):
        print(fr'update_chat_display 들어옴self.messages :{self.messages}')
        # 메시지 출력
        self.chat_display.clear()

        # 이전 메시지의 역할을 저장할 변수
        previous_role = None

        for message in self.messages:
            role = message["role"]
            focus = message["focus"]
            content = message["content"]

            # role이 중복되면 표시하지 않음
            if role == previous_role:
                display_role = ""  # 중복된 경우 role은 빈 문자열로
            else:
                display_role = role.capitalize()

            # 다음 반복에서 role 중복 체크를 위해 현재 role을 저장
            previous_role = role

            # 아이콘 이미지 파일 경로 설정 및 크기 조정
            icon_path = "bot.png" if role == "assistant" else "team.png"
            icon_path_role = "bot_resized2.png" if role == "assistant" else "team_resized2.png"

            #icon_path_focus = "bot_resized2.png" if focus == "assistant" else "team_resized2.png"
            #focus: Literal["web", "academic", "video", "math"]
            if focus == "web":
                icon_path_focus = "web-link_resized2.png"
            elif focus == "academic":
                icon_path_focus = "knowledge_resized2.png"
            elif focus == "video":
                icon_path_focus = "find_resized2.png"  # 예시: admin에 대한 이미지
            elif focus == "math":
                icon_path_focus = "search_resized2.png"  # 예시: user에 대한 이미지
            else:
                icon_path_focus = "web_resized2"  # 기본값, 만약 focus가 다른 값이면 사용

            icon_html = f'<img src="{icon_path_role}" alt="{role}" style="width:0.004em;height:0.004em;vertical-align:middle;margin-right:0.3em;">'
            # 아이콘 크기를 24px로 설정
            icon_html_role = f'<img src="{icon_path_role}" alt="{role}" style="width:24px;height:24px;vertical-align:middle;margin-right:0.3em;">'
            icon_html_focus = f'<img src="{icon_path_focus}" alt="{focus}" style="width:24px;height:24px;vertical-align:middle;margin-right:0.3em;">'

            if role == "user":
                # 사용자 메시지는 오른쪽에 배치
                chat_message = f'<div style="text-align:right; padding-right:10px; margin-bottom:10px; clear:both;">{icon_html_role}: {content}</div>'
            else:
                # 어시스턴트 메시지는 왼쪽에 배치
                chat_message = f'<div style="text-align:left; padding-left:10px; margin-bottom:10px; clear:both;">{icon_html_role} ({icon_html_focus}): {content}</div>'


            # 메시지 내용 추가
            self.chat_display.append(chat_message)

            global global_chat_display  # 전역 변수를 사용한다고 명시
            chat_display = self.chat_display
            global_chat_display = chat_display  # 전역 변수에 저장
            print(fr'global_chat_display : {global_chat_display}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatApp()
    ex.show()
    sys.exit(app.exec_())

