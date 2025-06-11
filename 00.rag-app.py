import os
from dotenv import load_dotenv
from openai import AzureOpenAI


def main():
    # 콘솔 화면에 남아있는 이력 지우기
    os.system('cls' if os.name == 'nt' else 'clear')

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

    # 프롬프트 초기화(시스템 메시지로 역할부여)
    prompt = [
        {
            "role": "system",
            "content": "You are a travel assistant that provides information on travel service available from Margie's Travel Agency. "
        }
    ]

    # 사용자 입력을 받아서 프롬프트에 추가
    while True:
        # 사용자 입력 받기
        input_text = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")

        # 종료 조건 처리
        if input_text.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        elif input_text.strip() == "":
            print("질문을 입력하세요.")
            continue
        
        # 프롬프트에 사용자 입력 추가
        prompt.append({"role": "user", "content": input_text})

        # RAG 파라미터 설정
        rag_params = {
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": { # Azure Search 파라미터 설정
                        "endpoint": search_endpoint,
                        "index_name": index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": search_api_key,
                        },
                        "query_type": "vector", # 벡터 검색을 사용 (text, vector, hybrid 중 선택 가능)
                        "embedding_dependency": { # 임베딩 모델 의존성 설정
                            "type": "deployment_name",
                            "deployment_name": embedding_model
                        }
                    },
                }
            ]
        }
    
        # Azure OpenAI에 프롬프트와 RAG 파라미터를 사용하여 응답 생성
        response = chat_client.chat.completions.create(
            model=chat_model,
            messages=prompt,
            extra_body=rag_params
        )

        # 응답에서 생성된 메시지 추출 및 출력
        completion = response.choices[0].message.content
        print(completion)

        # 프롬프트에 응답 추가
        prompt.append({"role": "assistant", "content": completion})

if __name__ == "__main__": # 이 파일이 직접 실행될 때 main() 함수를 호출
    main()