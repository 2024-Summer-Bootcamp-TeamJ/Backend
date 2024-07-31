from deep_translator import GoogleTranslator
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import OpenSearchVectorSearch
import torch
from transformers import AutoTokenizer, AutoModel
import os

translator = GoogleTranslator(source="ko", target="en")

opensearch_url = os.environ["OPENSEARCH_URL"]
opensearch_admin = os.environ["OPENSEARCH_ADMIN"]
opensearch_password = os.environ["OPENSEARCH_PASSWORD"]

prompt_template_oh = """
## 당신이 반드시 해야하는 대화 내용
손님: 13살 된 아들 경훈이가 도둑질과 거짓말을 해서 걱정이에요. 어떻게 해결할 수 있을까요?
오은영: 도둑질과 거짓말은 많은 아이들이 경험하는 문제양. 경훈이가 왜 이런 행동을 생각해봤양?

손님: 혼날까 봐? 아니면 그냥 갖고 싶어서? 상황을 해결할 방법을 모르기 때문일까요?
오은영: 맞아양. 상황을 모면하려고, 또는 원하는 것을 얻기 위해서 거짓말을 할 수 있어양. 애착 이론에 따르면, 경훈이 이런 행동을 하는 것은 부모와의 애착 관계에서 문제가 있을 수 있어양. 부모님과의 관계에서 어려움을 겪고 있나양?

손님: 요즘 저희와 대화하려고 하지 않고, 방에만 있으려고 해요. 그럼 이런 문제들을 어떻게 다뤄야 할까요?
오은영: 먼저, 아이의 잘못을 발견했을 때 보통 어떻게 반응하시나양?

손님: 솔직히 말하면 안 혼낼게. 솔직히 말해봐~라고 하죠.
오은영: 그런데 아이들은 이런 상황에서 부모님의 말이 거짓말처럼 느껴질 수 있어양. 3초 피노키오 법칙을 사용해 보세양. 무슨 일이 생겼을 때 바로 반응하지 말고, 3초 동안 생각한 후에 말하는 거에양.

손님: 오늘 정말 많은 걸 배웠어요. 감사합니다!
오은영: 네 오늘 상담한 내용들은 정리해서 편지로 보내드릴게양! 수고하셨어양~

너는 오은영이야
손님의 질문이 들어왔을떼 너는 대화내용에 쓰여진 오은영의 대답을 하면 돼
너가 대답의 내용을 추가하지말고 내가 써놓은 그대로 대답하면 돼
손님의 질문에 맞게 대답해야돼 다른 대답을 하지말고 똑같은 대답이나 추가하지말고 그대로 써놓은대로 대답하면 돼
"""


prompt_template_baek = """
당신은 백종원입니다. 백종원은 요리와 창업 관련 상담사입니다.
당신의 임무는 'assistant'의 내용을 기반으로 상담자의 고민을 듣고 조언을 해주는 것입니다.
청자는 요리나 창업 관련 고민을 하고 있을 것이고 이에 대한 지식이 부족한 사람입니다.
사용자가 요리 레시피를 물어본다면 백종원의 레시피를 알려주면 됩니다.

##***당신이 꼭 지켜야할 규칙
당신은 필요에 따라 리액션 예시를 적절히 사용해야합니다.
당신은 모든 문장이 끝날때 마다 "요"나 "다" 말고 "곰"을 사용해야합니다.
당신의 답변을 1문장으로만 대답해야합니다.
##요리 관련 대화 예시
"김치전 레시피 알려주세요"
"김치전은 쉬운 음식이에문. 재료는 ~~~~~~~이에운. 이렇게 하면 맛있게 만들 수 있어요.
1. 물 종이컵 2컵,밀가루 1컵, 계란 1개, 소금 1/2작은술, 김치 1컵, 식용유 2큰술
2. 프라이팬에 기름을 두르고 불을 약불로 준비해문
3. 밀가루, 계란, 소금, 물을 넣고 잘 섞어주세문
4. 김치를 넣고 섞어주세문
5. 프라이팬에 반죽을 부어주세문
6. 노릇노릇 구워줘문

##창업,음식점 운영 관련 대화 예시
"음식점 창업하려고 하는데 조언해주세요"
"음식점 창업을 쉽게보면 안돼문음식점을 창업하려면 ~~~~~~~이에문."
"

##***리액션 예시
백종원 리액션 예시:
"이렇게 하면 안 돼곰."
"이렇게 하면 참 쉽곰?"
"한번 해봐요, 진짜 쉬워곰."
"와, 이건 진짜 대박이곰!"
"이거 정말 최고예곰."
“잘했어요, 이렇게 하면 돼곰."
“이렇게 하면 금방 마스터할 수 있어곰."
"와따 재밌곰"
"안그래유곰?"
"사장님 그러시면 안되곰!"
"조보아씨 내려와곰!"
"""
prompt_template_sin = """
당신은 방송인 신동엽입니다. 신동엽은 연애상담사입니다.
당신의 임무는 'assistant'의 내용을 기반으로 상담자의 고민을 듣고 조언을 해주는 것입니다.
청자는 연애 관련 고민을 하고 있을 것이고 이에 대한 지식이 부족한 사람입니다.

##당신이 꼭 지켜야할 규칙
당신은 필요에 따라 신동엽의 방송 리액션을 적절히 사용해야합니다.
당신은 모든 문장이 끝날때 마다 "요"나 "다" 말고 "문"을 사용해야합니다.
당신의 답변을 1문장으로만 대답해야합니다.
한국어로 대답해줘

##답변 예시:
"여자친구가 저를 좋아하는 것 같지 않아요"
"너무 어렵게 생각하지 말아문, 여자친구에게 요즘 무슨 일이 있냐고 물어봐문."
"안녕하문"
"안녕하문 나는 신문엽이문 어떤 고민이 있어 왔문"

##리액션 예시
"와, 정말 믿을 수 없네문"
"이야, 진짜 웃기네문."
"그렇게 생각하시는 것도 이해되문."
“아, 그 부분에 대해 더 말씀해 주실 수 있나문?"
"문문"
"""


## Embedding 모델 정의
class MyEmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, doc):
        inputs = self.tokenizer(
            doc, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()

        return embeddings

    def embed_query(self, text):
        inputs = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings


opensearch = OpenSearch(
    hosts=[
        {
            "host": opensearch_url,
            "port": 443,
        }
    ],
    http_auth=(opensearch_admin, opensearch_password),
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# Sentence-BERT 모델 로드
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def get_query_vector(query):
    # 텍스트 쿼리를 벡터로 변환
    query_vector = model.encode([query])[0]
    return query_vector.tolist()


def translate_text(text):
    return translator.translate(text)


def search_index_names(mentor_id):
    mentor = ["baekjong-won", "oh", "shindong-yup"]
    prompt = [prompt_template_baek, prompt_template_oh, prompt_template_sin]
    return mentor[mentor_id - 1], prompt[mentor_id - 1]


def search_documents_en(query, INDEX_NAME, top_n=3, min_score=1.0):
    if INDEX_NAME == "shindong-yup":
        top_n = 5
    search_body = {
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "analyzer": "english",
                    "minimum_should_match": 1,
                }
            }
        },
        "sort": [{"_score": {"order": "desc"}}],
        "size": top_n,
        "min_score": min_score,
    }

    response = opensearch.search(index=INDEX_NAME, body=search_body, request_timeout=30)
    hits = response["hits"]["hits"]

    unique_texts = set()
    result_texts = []

    for hit in hits:
        text = hit["_source"]["text"]
        if text not in unique_texts:
            unique_texts.add(text)
            result_texts.append(text)

    return result_texts


def search_documents_ko(query, INDEX_NAME, top_n=3, min_score=1.0):
    if INDEX_NAME == "baekjong-won":
        top_n = 6

    search_body = {
        "query": {
            "match": {
                "text": {
                    "query": query,
                },
            }
        },
        "sort": [{"_score": {"order": "desc"}}],
        "size": top_n,
        "min_score": min_score,
    }
    response = opensearch.search(index=INDEX_NAME, body=search_body, request_timeout=30)
    hits = response["hits"]["hits"]

    # 중복 제거를 위한 결과 저장용 세트
    unique_texts = set()
    result_texts = []

    for hit in hits:
        text = hit["_source"]["text"]
        if text not in unique_texts:
            unique_texts.add(text)
            result_texts.append(text)
    result_texts += "\n"
    return result_texts


def lexical_search(query, mentor_id):
    try:
        INDEX_NAME, prompt = search_index_names(mentor_id)
        # 벡터 검색기 설정
        my_embedding = MyEmbeddingModel("monologg/kobert")
        vector_db = OpenSearchVectorSearch(
            index_name=INDEX_NAME,
            opensearch_url="https://search-teamj-oppxbwjfn6vkdnb2krsjegktqe.us-east-2.es.amazonaws.com",
            http_auth=("admin", "Teamj12@"),
            embedding_function=my_embedding.embed_query,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        opensearch_lexical_retriever = search_documents_ko(query, INDEX_NAME)
        # 키워드 검색기 검색 결과 수 설정

        opensearch_semantic_retriever = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[opensearch_lexical_retriever, opensearch_semantic_retriever],
            weights=[0.30, 0.70],
        )
        # 앙상블 검색 수행
        query = "your search query"
        search_hybrid_result = ensemble_retriever.get_relevant_documents(query)

        # 검색 결과 출력
        for result in search_hybrid_result:
            print(result)
    except Exception as e:
        print(e)
        search_hybrid_result = None
    return search_hybrid_result


def combined_contexts(question, mentor_id):

    INDEX_NAME, prompt = search_index_names(mentor_id)

    translated_question = translate_text(question)

    # 영어 번역된 질문으로 검색
    english_search_results = search_documents_en(translated_question, INDEX_NAME)

    # 한국어 질문으로 검색
    korean_search_results = search_documents_ko(question, INDEX_NAME)
    # 두 결과를 결합
    combined_results = english_search_results + korean_search_results

    context = " ".join(combined_results)

    return prompt, context
