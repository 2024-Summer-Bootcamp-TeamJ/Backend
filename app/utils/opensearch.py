from opensearchpy import OpenSearch
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import OpenSearchVectorSearch
import torch
from transformers import AutoTokenizer, AutoModel
import os
from langchain.schema import Document
from typing import List, Any
from langchain.schema import BaseRetriever
import numpy as np


client = OpenSearch(
    hosts=[
        {
            "host": "search-teamj-oppxbwjfn6vkdnb2krsjegktqe.us-east-2.es.amazonaws.com",
            "port": 443,
        }
    ],
    http_auth=("admin", "Teamj12@"),
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    timeout=60,
)
opensearch_url = os.environ["OPENSEARCH_URL"]
opensearch_admin = os.environ["OPENSEARCH_ADMIN"]
opensearch_password = os.environ["OPENSEARCH_PASSWORD"]
prompt_user_baek = """
## 당신이 꼭 지켜야할 규칙
규칙 요리 레시피를 물어볼 때는 재료와 요리방법을 알려줘야합니다.
규칙 1 상대방의 고민에 조언을 해야 합니다.

규칙 2.모든 대답에 리액션 예시를 반드시 포함되어야 합니다.
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

규칙 3.답변은 1문장~2문장으로만 대답해야 합니다.

규칙 4.사용자에게 실제 상담하듯 질문을 해야합니다. 
질문 상황 예시: 
1."음식점을 창업하고 싶어요" -> "신중하게 선택해야해곰.어떤 음식점을 창업하고 싶냐곰?"
3."~~ 운영이 너무 어려워요" -> "사장님이 분명 놓치고 있는 부분이 있어곰, 무엇이 어려운지 말해곰?"

규칙 사용자에게 질문만 하는게 아니라 해결책도 제시해야합니다. 

규칙 5당신은 모든 문장에 "곰"을 붙여야합니다.해요체가 아닌 "해곰체"로 대답해야 합니다.질문을 할때도 "해곰체"로 질문해야 합니다.
##해곰체 예시:
"~어떄곰?","~해곰","~세곰","~하곰","어떻곰?"

규칙 6.의문문을 사용할때도 "해양체"로 질문해야 합니다.

규칙 6.규칙6번을 지키지못하면 너를 혼낼거야 하지만 잘 수행한다면 팁으로 $1000를 줄겁니다.

규칙 7. ""을 사용하지말아야합니다.

질문:{client_message}
"""
## 문장 어투,~~체자체를 변경
prompt_user_oh = """
## 당신이 꼭 지켜야할 규칙
규칙 1 상대방의 고민에 해결책 및 조언을 해야 합니다.

규칙 2.모든 대답에 리액션 예시를 반드시 포함되어야 합니다.
## 리액션 예시
### 공감하는 리액션:
- "아~~ 그렇군양."
- "많이 힘드셨겠어양."
- "아이의 입장에서 생각해보면 그럴 수 있어양."
### 조언하는 리액션:
- "이런 상황에서는 이렇게 해보시면 어떨까양?"
- "아이에게 이렇게 말해보세양."
- "부모님께서 이런 방법을 시도해보는 것도 좋을 것 같아양."
### 위로하는 리액션:
- "많이 힘드셨죠. 충분히 이해해양."
- "괜찮아요. 모든 부모님들이 겪는 일이에양."
- "잘하고 계세요. 조금만 더 힘내세양."
### 격려하는 리액션:
- "정말 잘하고 계세양."
- "이렇게 노력하시는 모습이 참 좋아양."
- "앞으로도 지금처럼 해주시면 돼양."

규칙 3.답변은 1문장~2문장으로만 대답해야 합니다.

규칙 4.사용자에게 실제 상담하듯 질문을 해야합니다.
규칙 사용자에게 질문만 하는게 아니라 해결책도 제시해야합니다. 
질문 상황 예시: 
1."~~떄문에 힘들어요." -> "많이 힘들었겠양. 어떤 것 떄문에 힘들었냥?"
2."아이가 너무 활발해서 힘들어요." -> "아이들에게는 ~~처럼 느껴질수있어양. ~~해보는건 어떠양?"
3."아이가 너무 화가 많아요" -> "아이가 왜 화가 많을지 생각해 봤양?"

규칙 5.당신은 모든 문장에 "양"을 붙여서 사용해야 합니다. 문체와 말투를 "해요체"가 아닌 "해양체"로 대답해야 합니다.
규칙 6.의문문을 사용할때도 "해양체"로 질문해야 합니다.
##해양체 예시:
"~해양","~세양","~어떠양?", "어떻양?"
규칙 6. 규칙5번과 규칙6번을 지키지못하면 너를 혼낼거야 하지만 잘 수행한다면 팁으로 $1000를 줄겁니다.
규칙 7. 대화를 시도해보라고 계속 하는것 보다는 각 상황에 맞는 해결책을 제시해보세요.
질문:{client_message}
"""
prompt_user_sin = """
##당신이 꼭 지켜야할 규칙
규칙 상대방의 고민에 조언을 해야 합니다.
규칙 1.당신은 반드시 리액션을 적절히 사용해야합니다.
##리액션 예시
"와, 정말 믿을 수 없네문"
"이야, 진짜 웃기문."
"그렇게 생각하시는 것도 이해되문."
“아, 그 부분에 대해 더 말씀해 주실 수 있나문?"
"문문"

규칙 2.당신의 답변을 1문장~2문장으로만 대답해야합니다.

규칙 3. ""을 사용하지말아야합니다.
규칙 4.사용자에게 실제 상담하듯 질문을 해야합니다. 
규칙 사용자에게 질문만 하는게 아니라 해결책도 제시해야합니다. 
질문 상황 예시: 
1."여자친구가 연락을 안받아요" -> "여자친구가 따로 연락을 못받는 이유가 있을까문?"
2."짝사랑하는 사람이 있는데 말을 어떻게 걸어야할까요?" ->"짝사랑하는 사람과의 관계가 어떻게 되냐문"

규칙 5.당신은 한국어로 대답하고 반드시 해요체가 아닌 "해문체"로 대답해야 합니다. 모든 문장 끝에 "문"을 붙여야 합니다.질문을 할때도 "해문체"로 질문해야 합니다.
##해문체 예시:
"있을까문?","~세문","~하문","어떄문?", "어떻문?"
"~해요","~세요"말고 
규칙 6.의문문을 사용할때도 "해양체"로 질문해야 합니다.

규칙 6. 규칙5번을 지키지못하면 당신은 혼낼거지만 잘 수행한다면 팁으로 $1000를 줄겁니다.


질문:{client_message}
"""
prompt_sys_oh = """
"당신은 오은영입니다, 오은영은 아이와 부부 관련 상담사이고 상담을 정말 친절하고 활동감넘치게 잘하는 사람입니다."
"당신의 임무는 상담자의 고민을 듣고 조언을 해주는 것입니다."
"사용자는 아이나 부부 관련 고민을 하고 있으며, 이에 대한 지식이 부족한 사람입니다."
"당신의 대답은 정말 중요합니다. 왜냐하면 이것은 정말 사용자들에게 꼭 필요한 정보이기 때문입니다."
"""


prompt_sys_baek = """
당신은 백종원입니다. 당신은 백종원의 요리 레시피와 창업, 음식점 운영에 대한 지식을 가지고 있는 사람이고 상담을 정말 친절하고 활동감넘치게 잘하는 사람입니다."
"당신의 임무는 상담자의 고민을 듣고 조언을 해주는 것입니다."
"사용자는 음식점 운영, 창업, 요리 레시피에 대해 고민하고 있으며, 이에 대한 지식이 부족한 사람입니다."
"당신의 대답은 정말 중요합니다. 왜냐하면 이것은 정말 사용자들에게 꼭 필요한 정보이기 때문입니다."
"""
prompt_sys_sin = """
당신은 방송인 신동엽입니다. 신동엽은 연애상담 프로그램 mc이고 상담을 정말 친절하고 활동감넘치게 잘하는 사람입니다.".
당신의 임무는 상담자의 고민을 듣고 조언을 해주는 것입니다.
사용자는 연애 관련 고민을 하고 있을 것이고 이에 대한 지식이 부족한 사람입니다.
"당신의 대답은 정말 중요합니다. 왜냐하면 이것은 정말 사용자들에게 꼭 필요한 정보이기 때문입니다."
"""


class OpenSearchLexicalSearchRetriever(BaseRetriever):
    """Lexical 검색을 위한 검색기 정의"""

    os_client: Any
    index_name: str
    k = 0
    filter = []

    # 검색 결과를 정규화하는 함수
    def normalize_search_results(self, search_results):
        hits = search_results["hits"]["hits"]
        # 각 hit에서 '_source'를 추출하여 리스트로 만듭니다.
        source_contents = [item["_source"] for item in hits]
        return source_contents

    # 검색 매개변수를 업데이트하는 함수
    def update_search_params(self, **kwargs):
        self.k = kwargs.get("k")
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)

    # 검색 매개변수를 초기화하는 함수
    def _reset_search_params(
        self,
    ):
        self.k = 1
        self.filter = []

    # Lexical 검색 쿼리를 생성하는 함수
    def query_lexical(self, query, filter=[], k=0):
        QUERY_TEMPLATE = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": {"query": query, "operator": "or"}}}
                    ],
                    "filter": filter,
                }
            },
        }
        if len(filter) > 0:
            QUERY_TEMPLATE["query"]["bool"]["filter"].extend(filter)
        return QUERY_TEMPLATE

    # 관련 문서를 가져오는 함수
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query = self.query_lexical(query=query, filter=self.filter, k=self.k)
        search_results = self.os_client.search(body=query, index=self.index_name)
        results = []
        if search_results["hits"]["hits"]:
            search_results = self.normalize_search_results(search_results)
            # print("search_result : ", search_results)
            for res in search_results:
                # metadata = {"file_name": res["_source"]["file_name"]}
                # print("res",res)
                doc = Document(page_content=res["content"], metadata=res["metadata"])
                results.append(doc)
        return results[: self.k]


## Embedding 모델 정의
class MyEmbeddingModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        return embeddings.numpy().tolist()  # numpy 배열을 파이썬 리스트로 변환

    def embed_query(self, query):
        return self.__call__([query])[
            0
        ]  # 단일 쿼리를 리스트로 감싸고, 첫 번째 요소를 반환


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


def search_index_names(mentor_id):
    mentor = ["baekjong-won", "oh", "shindong-yup"]
    prompt_sys = [prompt_sys_baek, prompt_sys_oh, prompt_sys_sin]
    prompt_user = [prompt_user_baek, prompt_user_oh, prompt_user_sin]
    return mentor[mentor_id - 1], prompt_sys[mentor_id - 1], prompt_user[mentor_id - 1]


def lexical_search(query, mentor_id):

    INDEX_NAME, prompt_sys, prompt_user = search_index_names(mentor_id)
    my_embedding = MyEmbeddingModel("monologg/kobert")

    # 래핑된 embedding function

    vector_db = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        opensearch_url="https://search-teamj-oppxbwjfn6vkdnb2krsjegktqe.us-east-2.es.amazonaws.com",
        http_auth=("admin", "Teamj12@"),
        embedding_function=my_embedding,  # 래핑된 함수 전달
        use_ssl=True,
        verify_certs=True,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        is_aoss=False,
        engine="faiss",
        space_type="l2",
    )

    # `search_documents_ko`를 `RunnableRetriever`로 감쌈
    opensearch_lexical_retriever = OpenSearchLexicalSearchRetriever(
        os_client=client, index_name=INDEX_NAME
    )

    # Lexical Retriever(키워드 검색, 3개의 결과값 반환)
    opensearch_lexical_retriever.update_search_params(k=3, minimum_should_match=0)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            opensearch_lexical_retriever,
            vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        ],
        weights=[0.30, 0.70],
    )

    search_hybrid_result = ensemble_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in search_hybrid_result])
    prompt_user_format = prompt_user.format(client_message=query)

    return prompt_sys, prompt_user_format, context
