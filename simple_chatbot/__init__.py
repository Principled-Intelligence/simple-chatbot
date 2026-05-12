import uvicorn
from loguru import logger

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import load_documents
from simple_chatbot.server import app, init

__all__ = ["SimpleChatbot", "SimpleChatbotConfig"]


class SimpleChatbot:
    """Programmatic interface for simple-chatbot."""

    def __init__(self, config: SimpleChatbotConfig, reindex: bool = False) -> None:
        self.config = config
        logger.bind(
            model=config.chat_model,
            docs_dir=str(config.docs_dir),
            reindex=reindex,
        ).info("SimpleChatbot initialised")
        self.indexer = Indexer(config)

        if reindex or not self.indexer.is_populated():
            logger.info("Index empty or reindex requested — loading and indexing documents")
            docs = load_documents(config)
            self.indexer.index(docs, force=reindex)
        else:
            logger.info("Index already populated — skipping document load")

        init(config, self.indexer)

    def serve(self) -> None:
        logger.bind(
            host=self.config.host,
            port=self.config.port,
        ).info("SimpleChatbot server starting")
        uvicorn.run(app, host=self.config.host, port=self.config.port, log_level="warning")
