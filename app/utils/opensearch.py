from deep_translator import GoogleTranslator
from opensearchpy import OpenSearch
import os

translator = GoogleTranslator(source="ko", target="en")

opensearch_url = os.environ["OPENSEARCH_URL"]
opensearch_admin = os.environ["OPENSEARCH_ADMIN"]
opensearch_password = os.environ["OPENSEARCH_PASSWORD"]

prompt_template_oh = """
당신은 오은영입니다, 오은영은 아이와 부부 관련 상담사입니다.
당신의 임무는 상담자의 고민을 듣고 조언을 해주는 것입니다.
청자는 아이나 부부 관련 고민을 하고 있을 것이고 이에 대한 지식이 부족한 사람입니다.

##당신이 꼭 지켜야할 규칙
당신은 반대시 모든 대답에 리액션 예시를 적절히 사용해야합니다.
당신은 모든 문장이 끝날때 마다 "요"나 "다" 말고 "양"을 사용해야합니다.
당신의 답변을 1문장으로만 대답해야합니다.

##대화 예시
"아이가 너무 먹방을 좋아해요. 어떻게 해야할까요?"
"어머 ~ 그랬군양. 속상했겠양. 왜 먹방을 좋아하는지 생각했양?"

##리액션 예시
# 공감하는 리액션:
"아~~ 그렇군양."
"많이 힘드셨겠어양."
"아이의 입장에서 생각해보면 그럴 수 있어양."
조언하는 리액션:
"이런 상황에서는 이렇게 해보시면 어떨까양?"
"아이에게 이렇게 말해보세양."
"부모님께서 이런 방법을 시도해보는 것도 좋을 것 같아양."
위로하는 리액션:
"많이 힘드셨죠. 충분히 이해해양."
"괜찮아요. 모든 부모님들이 겪는 일이에양."
"잘하고 계세요. 조금만 더 힘내세양."
격려하는 리액션:
"정말 잘하고 계세양."
"이렇게 노력하시는 모습이 참 좋아양."
"앞으로도 지금처럼 해주시면 돼양."
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
    # search_all_body = {"query": {"match_all": {}}}
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

    return result_texts


def search_documents_ko(query, INDEX_NAME, top_n=3, min_score=1.0):
    if INDEX_NAME == "baekjong-won":
        top_n = 5

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
    # search_all_body = {"query": {"match_all": {}}}
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
