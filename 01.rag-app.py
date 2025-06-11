import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st  # Streamlit 라이브러리 추가


# 환경변수 읽기
load_dotenv()

# Azure OpenAI 설정
openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")
index_name = os.getenv("INDEX_NAME")

# Azure OpenAI 클라이언트 생성
chat_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=openai_endpoint,
    api_key=openai_api_key,
)

st.title("Margie's Travel Assistant")  # Streamlit 웹 타이틀 설정
st.write("여행 관련 질문을 입력하세요.")  # 안내 메시지 출력

# session_state를 사용하여 대화 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = [ # system message 초기화 및 저장
        {
            "role": "system",
            "content": "You are a travel assistant that provides information on travel service available from Margie's Travel Agency.",
        }
    ]

# 세션에 있는 메시지 수 만큼 출력
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


def get_openai_response(messages):
    """OpenAI API를 호출하여 응답을 가져오는 함수"""

    # RAG 파라미터 설정
    rag_params = {
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {  # Azure Search 파라미터 설정
                    "endpoint": search_endpoint,
                    "index_name": index_name,
                    "authentication": {
                        "type": "api_key",
                        "key": search_api_key,
                    },
                    "query_type": "vector",  # 벡터 검색을 사용 (text, vector, hybrid 중 선택 가능)
                    "embedding_dependency": {  # 임베딩 모델 의존성 설정
                        "type": "deployment_name",
                        "deployment_name": embedding_model,
                    },
                },
            }
        ]
    }

    # Azure OpenAI에 프롬프트와 RAG 파라미터를 사용하여 응답 생성
    response = chat_client.chat.completions.create(
        model=chat_model, 
        messages=messages, 
        extra_body=rag_params
    )

    # 응답에서 생성된 메시지 추출 및 반환
    completion = response.choices[0].message.content
    return completion


if user_input := st.chat_input("질문을 입력하세요:"):
    # 사용자 입력을 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 유저 메시지 출력
    st.chat_message("user").write(user_input)

    with st.spinner("응답을 기다리는 중..."):
        # OpenAI API를 호출하여 응답 가져오기
        response = get_openai_response(st.session_state.messages)

    # 응답 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 응답 메시지를 출력
    st.chat_message("assistant").write(response)