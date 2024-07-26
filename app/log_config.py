import os, logging
import logging.config
from formatter import ColoredFormatter


def setup_logging(config_file="log.ini"):
    logging.config.fileConfig(config_file)

    # 루트 로거 가져오기
    logger = logging.getLogger()

    # 기존 핸들러 가져오기
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            console_handler = handler
            break

    # 사용자 정의 포매터 설정
    if console_handler:
        console_handler.setFormatter(
            ColoredFormatter(
                "%(levelname)-5s - %(asctime)s - %(funcName)s - %(message)s"
            )
        )
    
    logger.setLevel(logging._nameToLevel[os.getenv("LOG_LEVEL")])

# setup_logging을 호출하여 로깅 설정 적용
if os.environ.get("TESTING") != "True":
    setup_logging()
