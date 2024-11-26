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

import ast
from langgraph.graph import END, StateGraph, START

from langchain_core.prompts import ChatPromptTemplate

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QTextEdit, QLineEdit, \
    QPushButton, QLabel


from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLineEdit, QLabel, QComboBox

import sys

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit

nest_asyncio.apply()
warnings.filterwarnings('ignore')
import speech_recognition as sr

 
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain.document_loaders import YoutubeLoader
memory = MemorySaver()
# 전역 변수로 QTextEdit 객체를 선언
global_text_edit = None  # 초기화 (전역 변수)
global m

# 메모리 세팅 시 필요한 키들
memory_config = {
    "thread_id": "your_thread_id",
    "checkpoint_ns": "your_namespace",
    "checkpoint_id": "your_checkpoint_id"
}

import tensorflow as tf
import numpy as np
# 새로운 주문이 시작될 때마다 order_details_list 초기화
order_details_list = []

# 각 아이스크림 주문 항목을 저장하는 order_details 초기화
order_details = {
    "flavor": None,     # 아이스크림 맛
    "toppings": []      # 추가 토핑 리스트
}

class IceCreamRecommender:
    def __init__(self):
        # 예시 데이터 (아이템과 사용자)
        self.users = ['user1', 'user2', 'user3']
        self.items = [
            '바나나맛_로투스', '바나나맛_레인보우', '바나나맛_오레오',
            '초코맛_로투스', '초코맛_레인보우', '초코맛_오레오',
            '딸기맛_로투스', '딸기맛_레인보우', '딸기맛_오레오'
        ]

        # 사용자-아이템 매핑을 인덱스로 변환
        self.user_to_index = {user: i for i, user in enumerate(self.users)}
        self.item_to_index = {item: i for i, item in enumerate(self.items)}

        # 구매 데이터 (1로 표시)
        self.purchase_interactions = [
            ('user1', '바나나맛_로투스'),
            ('user2', '초코맛_레인보우'),
            ('user3', '딸기맛_오레오'),
        ]

        # 장바구니 데이터 (0.5로 표시, 장바구니에 담은 아이템은 구매보다 낮은 가중치 부여)
        self.cart_interactions = [
            ('user1', '초코맛_오레오'),
            ('user2', '바나나맛_레인보우'),
            ('user3', '딸기맛_로투스'),
        ]

        # 사용자-아이템 상호작용을 인덱스로 변환 (구매는 1, 장바구니는 0.5)
        self.user_indices = []
        self.item_indices = []
        self.interaction_values = []

        # 구매 상호작용
        for user, item in self.purchase_interactions:
            self.user_indices.append(self.user_to_index[user])
            self.item_indices.append(self.item_to_index[item])
            self.interaction_values.append(1)

        # 장바구니 상호작용
        for user, item in self.cart_interactions:
            self.user_indices.append(self.user_to_index[user])
            self.item_indices.append(self.item_to_index[item])
            self.interaction_values.append(0.5)

        self.model = self.build_model()

    def build_model(self):
        user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="user")
        item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="item")

        embedding_dim = 8
        user_embedding = tf.keras.layers.Embedding(len(self.users), embedding_dim)(user_input)
        item_embedding = tf.keras.layers.Embedding(len(self.items), embedding_dim)(item_input)

        user_flattened = tf.keras.layers.Flatten()(user_embedding)
        item_flattened = tf.keras.layers.Flatten()(item_embedding)

        dot_product = tf.keras.layers.Dot(axes=1)([user_flattened, item_flattened])

        model = tf.keras.Model(inputs=[user_input, item_input], outputs=dot_product)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self):
        # 모델 학습
        self.model.fit([np.array(self.user_indices), np.array(self.item_indices)], np.array(self.interaction_values),
                       epochs=10)

    def recommend(self, user_id):
        # user_id를 인덱스로 변환
        if user_id not in self.user_to_index:
            raise ValueError("유효하지 않은 사용자 아이디입니다.")
        user_id_index = self.user_to_index[user_id]

        # 추천 시스템에서 아이템 예측
        scores = self.model.predict([np.array([user_id_index] * len(self.items)), np.arange(len(self.items))])
        top_items = np.array(self.items)[np.argsort(-scores.flatten())]
        return top_items[0]

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

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


def chatbot(state: State):
    llm_with_tools = llm.bind_tools(tools[state["focus"]])

    messages = filter_messages(state["messages"])
    result = llm_with_tools.invoke(messages)

    return {"messages": [result]}

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"

from langchain.document_loaders import WebBaseLoader
urls = [
            "https://blog.naver.com/silentsin/222284340647",
            "https://www.chocolate-academy.com/en/top-ice-cream-trends",
            "https://www.idfa.org/whats-hot-in-ice-cream/",

        ]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

from langchain.embeddings import OpenAIEmbeddings

# OpenAI 임베딩 객체 생성
embd = OpenAIEmbeddings(model="text-embedding-ada-002")
# Build vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()

docs = retriever.invoke(question)
#doc_txt = docs[1].page_content
doc_txt = docs[0].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# ChatApp 클래스 수정
# 전역 변수 선언
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

        send_button = QPushButton('음성 인식 시작', self)
        self.send_button = send_button

        send_button.clicked.connect(self.run_start_voice_recognition)

        layout.addWidget(send_button)
        self.send_button.setStyleSheet("background-color: #007BFF; color: white; font-size: 16px;")

        # 레이아웃 설정
        self.setLayout(layout)
        self.resize(400, 300)
        self.resize(800, 1000)

        # QTextEdit 스타일 (파란색 배경과 흰색 텍스트)
        self.chat_display.setStyleSheet("background-color: lightblue; color: darkblue; font-size: 16px;")

    def run_start_voice_recognition(self):
        while True:
            # 음성 인식 함수 호출

            my_speech = self.start_voice_recognition()

            # 종료 명령 처리
            if my_speech == "종료":
                print("음성 인식 종료")
                break

            # 음성 인식 결과 출력
            print('[고객님]')
            print(my_speech)

            print('---' * 30)
            print()

    def start_voice_recognition(self):


            recognizer = sr.Recognizer()
            with sr.Microphone() as source:  # 마이크 열기

                recognizer.adjust_for_ambient_noise(source)

                # 마이크 입력 받기
                audio = recognizer.listen(source)

            try:
                mySpeech = recognizer.recognize_google(audio, language='ko-KR', show_all=False)

                self.message_input.setText(mySpeech)  # 텍스트 박스에 음성 결과 표시

                self.send_message()

            except sr.UnknownValueError:
                print("Google 음성 인식이 오디오를 이해할 수 없습니다.")

                user_message = self.message_input.text()  # 메시지 입력란에서 텍스트 가져오기
                user_message = "아이스크림이 안 차가워"
                user_message = "바나나맛 주문할래"
                user_message = "추천해줘"
                user_message = "유행하는 아이스크림 맛 뭐야"

                mySpeech = user_message

                self.message_input.setText(mySpeech)  # 텍스트 박스에 음성 결과 표시

                self.send_message()

            except sr.RequestError as e:
                print("Google 음성 인식 서비스에서 결과를 요청할 수 없습니다.; {0}".format(e))

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

        print("---ROUTE QUESTION---")
        #self.text_edit.append("---ROUTE QUESTION---")
        QApplication.processEvents()

        current_text = global_user_message

        print(f'현재 텍스트: {current_text}')  # 현재 텍스트 출력

        question = current_text

        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt

        system = """당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
                        벡터 스토어에는 에이전트, 프롬프트 엔지니어링, 적대적 공격과 관련된 문서가 포함되어 있습니다.
                        이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그렇지 않으면 웹 검색을 사용하십시오.
                        문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오."""

        system = """
                    당신은 사용자 질문을 벡터 스토어나 웹 검색으로 라우팅하는 전문가입니다.
        벡터 스토어에는 판매데이터와 관련된 문서가 포함되어 있습니다.
        이러한 주제에 대한 질문은 벡터 스토어를 사용하십시오. 그 외의 경우에는 웹 검색을 기본 설정으로 사용하십시오.
        문서는 영어로 되어 있을 수 있지만, 한국어로 된 링크를 우선적으로 제공하고, 답변은 한글로 번역하여 제공하십시오.
 
                    """

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
            #QApplication.proceorder_intent_checkssEvents()

            QApplication.processEvents()

            return "vectorstore"

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        state["question"] = global_user_message  # question 값을 state에 저장
        state["documents"] = global_user_message  # documents 값을 state에 저장

        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:

            print(fr'grade_documents  : {d}')

            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            print(fr'grade_documents score : {score}')
            print(fr'grade_documents grade : {grade}')

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

                # self.text_edit.append("---GRADE: DOCUMENT RELEVANT---")
                print(fr'grade_documents filtered_docs : {filtered_docs}')
                QApplication.processEvents()
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # self.text_edit.append("---GRADE: DOCUMENT NOT RELEVANT---")
                QApplication.processEvents()
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        self.text_edit.append("---TRANSFORM QUERY---")
        QApplication.processEvents()
        question = state["question"]
        documents = state["documents"]

        print(fr'transform_query question : {question}')
        print(fr'transform_query  documents : {documents}')
        print(fr'transform_query  type(state) : {type(state)}')

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        print(fr'transform_query better_question : {better_question}')

        print("---better_question---")
        self.text_edit.append(question)
        QApplication.processEvents()

        return {"documents": documents, "question": better_question}

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print(fr'decide_to_generate state:{state}')

        print("---ASSESS GRADED DOCUMENTS---")
        self.text_edit.append("---ASSESS GRADED DOCUMENTS---")
        QApplication.processEvents()
        state["question"]
        filtered_documents = state["documents"]
        print(fr'decide_to_generate filtered_documents:{filtered_documents}')

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            self.text_edit.append("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            QApplication.processEvents()
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            self.text_edit.append("---DECISION: GENERATE---")
            QApplication.processEvents()
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print(fr'grade_generation_v_documents_and_question state:{state}')

        print("---CHECK HALLUCINATIONS---")
        self.text_edit.append("---CHECK HALLUCINATIONS---")
        QApplication.processEvents()
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        print(fr'grade_generation_v_documents_and_question question:{question}')
        print(fr'grade_generation_v_documents_and_question documents:{documents}')
        print(fr'grade_generation_v_documents_and_question generation:{generation}')

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        print(fr'grade_generation_v_documents_and_question score:{score}')
        grade = score.binary_score
        print(fr'grade_generation_v_documents_and_question grade:{grade}')
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            self.text_edit.append("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            QApplication.processEvents()
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            self.text_edit.append("---GRADE GENERATION vs QUESTION---")

            score = self.answer_grader.invoke({"question": question, "generation": generation})
            print(fr'grade_generation_v_documents_and_question score:{score}')

            grade = score.binary_score
            print(fr'grade_generation_v_documents_and_question grade:{grade}')

            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                print("---GRADE GENERATION vs QUESTION---")

                self.text_edit.append("---DECISION: GENERATION ADDRESSES QUESTION---")
                self.text_edit.append("---GRADE GENERATION vs QUESTION---")
                QApplication.processEvents()
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                self.text_edit.append("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                QApplication.processEvents()
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            self.text_edit.append("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            QApplication.processEvents()
            return "not supported"

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        state["question"] = global_user_message  # question 값을 state에 저장
        state["documents"] = global_user_message  # documents 값을 state에 저장
        print(fr'retrieve global_user_message :{global_user_message}')
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

        self.text_edit.append("--질문--")
        self.text_edit.append(question)
        self.text_edit.append("--답변--")
        self.text_edit.append(documents)
        QApplication.processEvents()

    def tools_condition(self, state):
        print(fr'tools_condition state :{state}')

        if state["messages"][-1].tool_calls:
            t = state['focus']

            return f"{state['focus']}_tools"
        return END

    def send_message(self):
        # 입력 메시지 가져오기
        user_message = self.message_input.text()

        global global_user_message  # 전역 변수를 사용한다고 명시
        user_message = self.message_input.text()  # 메시지 입력란에서 텍스트 가져오기
        global_user_message = user_message  # 전역 변수에 저장

        if user_message:

            selected_focus = self.get_selected_focus()  # self를 사용하여 get_selected_focus 호출

            print("send_message  Selected Focus:", selected_focus)
            # self.messages.append({"role": "assistant", "focus": selected_focus, "content": assistant_response})
            self.messages.append({"role": "user", "focus": selected_focus, "content": user_message})
            print(f'send_message  self.messages: {self.messages}')

            self.update_chat_display()

            # 예시 사용
            recommender = IceCreamRecommender()

            # 모델 학습
            #recommender.train_model()

            # 추천 아이템 받기
            #recommended_item = recommender.recommend('user1')

            # Build graph
            workflow = StateGraph(State)
            # workflow.add_node("chatbot", chatbot)

            for focus, tool_node in tool_nodes.items():
                workflow.add_node(f"{focus}_tools", tool_node)

            # workflow.add_node("web_search", web_search)  # web search
            workflow.add_node("web_search", chatbot)  # web search
            # workflow.add_node("retrieve", retrieve)  # retrieve
            workflow.add_node("retrieve", chatbot)  # retrieve
            # workflow.add_node("retrieve", retrieve)  # retrieve
            workflow.add_node("grade_documents", self.grade_documents)  # grade documents
            workflow.add_node("generate", self.generate)  # generate
            workflow.add_node("transform_query", self.transform_query)  # transform_query

            for focus in tools:
                workflow.add_edge(f"{focus}_tools", "web_search")

            workflow.add_conditional_edges(
                START,  # 시작 노드
                self.route_question,  # 발화 라우팅 함수
                {
                    "web_search": "web_search",  # 웹 검색 관련
                    "vectorstore": "web_search",  # 벡터 스토어 검색 관련
                    # "order_process": "order_process",  # 주문 관련 발화일 경우 order_process 실행
                }
            )

            workflow.add_conditional_edges(
                "web_search",
                self.tools_condition,
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
                self.decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "generate",
                },
            )
            workflow.add_edge("transform_query", "retrieve")
            workflow.add_conditional_edges(
                "generate",
                self.grade_generation_v_documents_and_question,
                {
                    "not supported": "generate",
                    "useful": END,
                    "not useful": "transform_query",
                },
            )

            #graph = workflow.compile(checkpointer=memory_config)

            graph = workflow.compile(checkpointer=memory)

            responses = []

            config2 = {
                "configurable": {
                    "thread_id": "2",  # thread_id 값 설정
                    "memory": self.memory  # 메모리 기능을 추가
                }
            }

            print(fr'send_message self.memory  : {self.memory }')

            selected_focus_tuple = tuple(selected_focus)
            selected_focus = selected_focus_tuple
            print(fr'send_message selected_focus : {selected_focus}')
            print(fr'send_message type(selected_focus) : {type(selected_focus)}')
            # If no search option is selected, default to web search
            if not selected_focus:
                selected_focus = ["web"]  # Set default to web if no option is selected

            selected_focus = tuple(selected_focus)  # 리스트를 튜플로 변환하여 사용

            if isinstance(selected_focus, list):
                selected_focus = tuple(selected_focus)

            for focus in selected_focus:
                print(fr' send_message focus : {focus}')

                from string import Template

                # 템플릿 정의
                template = Template("사용자의 질문은: $user_message\n현재 포커스는: $focus")

                # 템플릿을 사용하여 프롬프트 생성
                prompt = template.substitute(user_message=user_message, focus=focus)

                from string import Template

                system_message = """
                                 
                                당신은 사용자 질문에 대해 웹 검색을 통해 가장 관련성 높은 정보를 찾아 제공하는 전문가입니다.
                                검색 결과는 한국어로 된 웹사이트를 우선적으로 제공하고, 결과는 한글로 번역하여 답변합니다.

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
                                - 예: "주문하신 바나나맛 아이스크림에 오레오 토핑을 추가하여 1개 주문이 완료되었습니다. 감사합니다!"

                                ### 추천 접수 모드
                                사용자가 "추천해줘", "아이스크림 추천", "토핑 추천" 등 추천을 요청하는 발화가 있을 경우, 즉시 추천 함수를 실행하여 추천 결과를 바탕으로 응답합니다.  
                                함수 실행 후, **`recommended_item`**을 바탕으로 아래와 같은 형식으로 응답합니다:
                                - "추천해드리겠습니다! 잠시만 기다려 주세요. {recommended_item}을 추천드립니다!"
                                - "아이스크림 추천해드리겠습니다. 잠시만 기다려 주세요! {recommended_item} 중에서 선택해 보세요."
                                - "원하시는 토핑을 추천해드리겠습니다! {recommended_item} 중에서 선택해 보세요."
                                - "추천해드리겠습니다! 잠시만 기다려 주세요. {recommended_item}을 추천드립니다!"
                """

                global m
                # 대화 히스토리를 불러오고, 메모리에 이전 응답 저장
                m = self.memory.get(selected_focus, [])

                print(fr'm :{m}')
                # 사용자 메시지와 포커스를 기반으로 프롬프트 생성
                prompt = system_message + f"\n사용자 메시지: {user_message}\n현재 포커스: {focus}\n대화 히스토리: {m}"

                import re
                recommendation_pattern = r".*(추천).*"
                match_recommedation_pattern = re.search(recommendation_pattern, user_message)
                if match_recommedation_pattern:
                    recommended_item = recommender.recommend('user1')  # 사용자에게 추천

                    prompt = system_message + f"\n사용자 메시지: {user_message}\n현재 포커스: {focus}\n대화 히스토리: {m}\n추천 아이템: {recommended_item}"

                result = graph.invoke({
                    "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                    "focus": focus,
                }, config2)  # config2는 필요한 설정을 포함

                print(fr'send_message result: {result}')
                self.memory[selected_focus] = self.memory.get(selected_focus, []) + [user_message]

                m += self.memory.get(selected_focus, []) + [user_message, result]

                # 응답 처리
                if "messages" in result and len(result["messages"]) > 0:

                    print(f'send_message result : {result}')

                    messages = result["messages"]
                    print(f'send_message messages: {messages}')
                    assistant_response = result["messages"][-1].content

                    print(f'send_message assistant_response : {assistant_response}')

                    responses.append((selected_focus, assistant_response))

                self.messages.append({"role": "assistant", "focus": focus, "content": assistant_response})
                print(f'send_message  self.messages : {self.messages}')
                '''
                self.messages: [
    {'role': 'user', 'focus': [], 'content': '바나나맛 주문할래'}, 
    {'role': 'assistant', 'focus': 'web', 'content': '바나나맛 아이스크림을 주문하셨습니다. 추가할 토핑은 무엇으로 하시겠습니까? (로투스, 레인보우, 오레오 중에서 선택해 주세요.)'}]
                '''

                import re

                # self_messages = [
                #     {'role': 'user', 'focus': [], 'content': '바나나맛 주문할래'},
                #     {'role': 'assistant', 'focus': 'web',
                #      'content': '바나나맛 아이스크림을 주문하셨습니다. 추가할 토핑은 무엇으로 하시겠습니까? (로투스, 레인보우, 오레오 중에서 선택해 주세요.)'},
                #     {'role': 'assistant', 'content': '주문하신 바나나맛 아이스크림에 오레오 토핑을 추가하여 1개 주문이 완료되었습니다. 감사합니다!'}
                # ]

                # 정규식 패턴: 맛, 토핑, 개수 추출

                order_pattern = r"주문하신 (\S+) 아이스크림에 (\S+) 토핑을 추가하여 (\d+)개 주문이 완료되었습니다.*$"

                self_messages = self.messages
                # 'assistant' role만 확인
                for message in self_messages:

                    print(f'send_message  self.messages : {message}')

                    if message['role'] == 'assistant':
                        content = message['content']

                        print(
                            f'send_message  message content : {content}')

                        # 정규식으로 추출
                        match_order_pattern = re.search(order_pattern, content)

                        if match_order_pattern:
                            flavor = match_order_pattern.group(1)  # 아이스크림 맛
                            topping = match_order_pattern.group(2)  # 토핑
                            quantity = match_order_pattern.group(3)  # 개수
                            print(f"send_message  정규식으로 추출 아이스크림 맛: {flavor}, 토핑: {topping}, 개수: {quantity}")

                            # order_detail 딕셔너리 업데이트
                            order_detail = {
                                'flavor': flavor,
                                'topping': topping,
                                'quantity': quantity
                            }

                            # order_detail_list에 추가
                            order_details_list.append(order_detail)

                        # 결과 확인
                        print("주문 상세:", order_details)
                        print("주문 목록:", order_details_list)

                responses = []


                if "messages" in result and len(result["messages"]) > 0:
                    assistant_response = result["messages"][-1].content

                    print(f'send_message assistant_response: {assistant_response}')


                    assistant_response = {
                        "role": "assistant",
                        "focus": focus,  # focus 정보를 추가
                        "content": result["messages"][-1].content  # 응답 내용 추가
                    }

                    print(f'send_message assistant_response: {assistant_response}')
                    responses.append((selected_focus, assistant_response))
                    print(f'send_message responses: {responses}')

                print(f'send_message Updated responses: {responses}')
               
                # 메모리에 봇의 응답도 저장
                self.memory[selected_focus].append(assistant_response)
                print(f'send_message self.memory[selected_focus]: {self.memory[selected_focus]}')
                # UI 업데이트
                self.update_chat_display()

        # 입력 필드 초기화
        self.message_input.clear()

    def update_chat_display(self):

        # 메시지 출력
        self.chat_display.clear()

        # 이전 메시지의 역할을 저장할 변수
        previous_role = None

        for message in self.messages:
            role = message["role"]
            focus = message["focus"]
            content = message["content"]

            print(fr'update_chat_display role :{role}')
            print(fr'update_chat_display focus :{focus}')
            print(fr'update_chat_display content :{content}')

            # role이 중복되면 표시하지 않음
            if role == previous_role:
                display_role = ""  # 중복된 경우 role은 빈 문자열로
            else:
                display_role = role.capitalize()

            # 다음 반복에서 role 중복 체크를 위해 현재 role을 저장
            previous_role = role

            # 아이콘 이미지 파일 경로 설정 및 크기 조정
            #icon_path = "bot.png" if role == "assistant" else "team.png"
            icon_path_role = "bot_resized2.png" if role == "assistant" else "team_resized2.png"
            icon_path_role = "./resource/bot_resized2.png" if role == "assistant" else "./resource/team_resized2.png"
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
                icon_path_focus = "web-link_resized2.png"  # 기본값, 만약 focus가 다른 값이면 사용
            if focus == "web":
                icon_path_focus = "./resource/web-link_resized2.png"
            elif focus == "academic":
                icon_path_focus = "./resource/knowledge_resized2.png"
            elif focus == "video":
                icon_path_focus = "./resource/find_resized2.png"  # 예시: admin에 대한 이미지
            elif focus == "math":
                icon_path_focus = "./resource/search_resized2.png"  # 예시: user에 대한 이미지
            else:
                icon_path_focus = "./resource/web-link_resized2.png"  # 기본값, 만약 focus가 다른 값이면 사용

            #icon_html = f'<img src="{icon_path}" alt="{role}" style="width:15px;height:15px;vertical-align:middle;margin-right:5px;">'
            #글씨 크기에 맞게 아이콘 조정

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
            print(fr'update_chat_display global_chat_display : {global_chat_display}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatApp()
    ex.show()
    sys.exit(app.exec_())
 