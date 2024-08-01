from fastapi import APIRouter, Depends, WebSocket
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect
from langchain.memory import ConversationBufferMemory
from crud import user as UserService
from crud import chat as ChatService
from crud import chatroom as ChatroomService
from crud import prescription as PrescriptionService
from database import get_db
from utils import opensearch as opensearchService
from utils import celery_worker
from utils.gpt import get_gpt_answer
import re
import json
import tiktoken
import logging


logger = logging.getLogger(__name__)


router = APIRouter()


def count_tokens(text, model_name="cl100k_base"):
    tokenizer = tiktoken.get_encoding(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)


def generate_gpt_payload(prompt_user, chat_memory_messages, prompt_sys, context):
    gpt_payload = (
        [
            {"role": "system", "content": prompt_sys},  ##역할부여
            {"role": "assistant", "content": context},
        ]
        + [
            {"role": "assistant", "content": msg["content"]}
            for msg in chat_memory_messages
        ]
        + [{"role": "user", "content": prompt_user}]  ##규칙
    )

    combined_content = " ".join([message["content"] for message in gpt_payload])
    token_count = count_tokens(combined_content)

    # prompt 추가
    logger.debug(
        "Gpt Payload being Generated: prompt=%s, token count=%s",
        gpt_payload,
        token_count,
    )

    return gpt_payload


# 문자열을 한글, 영어, 숫자, 공백, 마침표, 쉼표, 물음표만 남기고 제거
def trim_text(text):
    trimmed_text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9 .,?]", "", text)
    logger.debug("Text Trimmed: text=%s", trimmed_text)
    return trimmed_text


@router.websocket("/chatrooms/{chatroom_id}")
async def websocket_endpoint(
    websocket: WebSocket, chatroom_id: int, user_id: int, db: Session = Depends(get_db)
):
    logger.info(
        "WebSocket Connection Requested: chatroom_id=%d, user_id=%d",
        chatroom_id,
        user_id,
    )
    await websocket.accept()
    chatroom = ChatroomService.get_chatroom(db, chatroom_id=chatroom_id)
    # chatroom이 없을 경우 연결 종료
    if chatroom is None:
        logger.info("Chatroom Not Found: chatroom_id=%d", chatroom_id)
        await websocket.send_json(
            {"event": "disconnect", "message": "Chatroom not found"}
        )
        await websocket.close()
        return

    # user_id와 chatroom_id가 일치하지 않을 경우 연결 종료
    user = UserService.get_user(db, user_id=user_id)
    if user is None:
        logger.info("User Not Found: user_id=%d", user_id)
        await websocket.send_json({"event": "disconnect", "message": "User not found"})
        await websocket.close()
        return

    # 현재 연결 정보 전송
    await websocket.send_json(
        {
            "event": "connect",
            "message": "connected",
            "user_id": user_id,
            "chatroom_id": chatroom_id,
            "mentor_id": chatroom.mentor_id,
        }
    )
    logger.info(
        "User Connected to Chatroom: user_id=%d, chatroom_id=%d", user_id, chatroom_id
    )

    memory = ConversationBufferMemory()

    try:
        while True:
            client_message = await websocket.receive_text()
            logger.info(
                "User Message Received: user_id=%d\nmessage=%s", user_id, client_message
            )
            # 사용자의 메시지를 db에 저장
            ChatService.create_chat(
                db, chatroom_id=chatroom_id, is_user=True, content=client_message
            )
            logger.info("User Message Saved: chatroom_id=%d", chatroom_id)

            # RAG 모델을 사용하여 prompt 생성

            prompt_sys, prompt_user, context = opensearchService.lexical_search(
                client_message, chatroom.mentor_id
            )

            logger.debug("Prompt Generated for Rag Model")

            # 대화 기록과 prompt를 합쳐서 전달할 payload 생성
            gpt_payload = generate_gpt_payload(
                prompt_user, memory.chat_memory.messages, prompt_sys, context
            )
            logger.info("Gpt Payload Generated: chatroom_id=%d", chatroom_id)

            # GPT에게 답변 요청
            try:
                gpt_answer = get_gpt_answer(gpt_payload)
            except Exception as e:
                logger.error("Gpt Answer Error: error=%s", e)
                await websocket.send_json(
                    {
                        "event": "server_message",
                        "message": "죄송합니다. 서버에 오류가 발생했습니다. 관리자에게 문의해주세요.",
                    }
                )
                continue

            logger.debug("Gpt Answer Received: answer=%s", gpt_answer)

            # 대화 기록에 사용자의 메시지와 GPT의 답변 추가
            memory.chat_memory.messages.append(
                {
                    "role": "assistant",
                    "content": client_message + gpt_answer,
                }
            )
            logger.debug(
                "Chat Memory Updated with User Message and Gpt Answer: user_message=%s, gpt_answer=%s",
                client_message,
                gpt_answer,
            )

            # 음성을 생성하는 celery task 실행
            server_audio = celery_worker.generate_audio_from_string.delay(
                trim_text(gpt_answer)
            ).get()
            logger.debug("Audio Generated with Gpt Answer")

            # GPT의 답변과 음성을 클라이언트에게 전송
            await websocket.send_json(
                {
                    "event": "server_message",
                    "message": gpt_answer,
                    "audio": server_audio,
                }
            )
            logger.info(
                "Answer and audio sent to user: user_id=%d, chatroom_id=%d\ngpt_answer=%s",
                user_id,
                chatroom_id,
                gpt_answer,
            )

            # GPT의 답변을 db에 저장
            ChatService.create_chat(
                db, chatroom_id=chatroom_id, is_user=False, content=gpt_answer
            )
            logger.info("Gpt Answer Saved: chatroom_id=%d", chatroom_id)

    # 연결이 끊어졌을 때
    except WebSocketDisconnect as e:
        logger.info(
            "WebSocket Disconected: user_id=%d, chatroom_id=%d", user_id, chatroom_id
        )

        prompt = """Create a brief letter-style summary that a counselor sends to a client in Korean based on the following conversation.
The summary should provide a concise solution derived from the conversation.
Limit the length of the summary to no more than a few sentences.
Conversation : """ + json.dumps(
            memory.chat_memory.messages, ensure_ascii=False
        )

        prescription_content = get_gpt_answer(
            [{"role": "assistant", "content": prompt}]
        )
        logger.debug("Prescription Generated: content=%s", prescription_content)

        # 모든 채팅 내용으로 처방전 생성
        prescription = PrescriptionService.create_prescription(
            db,
            user_id=user_id,
            mentor_id=chatroom.mentor_id,
            content=prescription_content,
        )
        logger.info(
            "Prescription Saved: user_id=%d, mentor_id=%d, prescription_id=%d",
            user_id,
            chatroom_id,
            prescription.id,
        )

        # 채팅방 삭제
        ChatroomService.delete_chatroom(db, chatroom_id=chatroom_id)
        logger.info("Chatroom deleted: chatroom_id=%d", chatroom_id)

        # 연결 종료
        logger.info("Client disconnected: user_id=%d, status_code=%s", user_id, e.code)
