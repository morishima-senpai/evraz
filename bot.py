import os
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command

from celery import Celery
import redis.asyncio as redis
from dotenv import load_dotenv

from src.embeddings import EmbeddingService
from src.workers import CodeReviewWorker, FileProcessReviewWorker

# Configure logging
logging.basicConfig(level=logging.INFO, filename="debug.log")
logger = logging.getLogger(__name__)


load_dotenv(".env")


# celery Configuration
celery_app = Celery(
    'evraz_code_review_bot', 
    broker = os.environ.get('REDIS_URL'), 
    backend = os.environ.get('REDIS_URL')
)
celery_app.autodiscover_tasks()

class CodeReviewBot:
    def __init__(self):
        
        # initialize Redis connection
        self.redis_client = redis.from_url(os.environ.get('REDIS_URL'))
        
        # initialize bot and dispatcher
        self.bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.dp = Dispatcher()
        
        # initialize services
        self.embedding_service = EmbeddingService(gen_embeedings=False)

        
        self.code_reviewer = CodeReviewWorker(
            llm_api_key=os.environ.get("LLM_API_KEY"),
            llm_base_url=os.environ.get("LLM_END_POINT"),
            model_name = "mistral-nemo-instruct-2407",
            vector_store_path = "./gen/",
            send_message_updates=self.send_message_updates,
            send_document_updates=self.send_document_updates
        )
        self.file_process_reviewer = FileProcessReviewWorker(
            code_reviewer=self.code_reviewer,
            send_message_updates=self.send_message_updates,
            send_document_updates=self.send_document_updates
        )
        # setup message handlers
        self._setup_handlers()

    def _setup_handlers(self):
        self.dp.message(Command('start'))(self.handle_start)
        self.dp.message(F.document)(self.handle_document)
        self.dp.message(F.file)(self.handle_document)
        self.dp.message(F.text)(self.handle_text_code)

    async def handle_start(self, message: Message):
        """Handle /start command"""
        await message.answer(
            "**🟢🟢 Здравствуйте, добро пожаловать в Evraz Code Review Bot😀** \n\n"
            "📎 Отправьте мне файл исходного кода или полный проект в виде `zip`-архива\n\n\n"
            "🗂️ Я могу прочитать практически любой исходный файл (`.js`, `.py`, `.cpp` итд.) и файлы `.zip`.\n\n"
            "Или вы также можете ввести код в виде сообщения.👀",
            parse_mode="markdown"
        )
        return


    async def handle_document(self, message: Message):
        """Handle code file uploads"""
        try:
            document = message.document
            
            # Download the file
            file = await self.bot.get_file(document.file_id)       
            
            if not document.file_name.endswith(".zip") :
                # submit document for async processing
                try:
                    downloaded_file = await self.bot.download_file(file.file_path)
                    await message.reply(f"Ух ты!! Все для проверки вашего кода! 🔥🔥")
                    
                    await self.process_code_review(
                        file_content=downloaded_file.read().decode('utf-8'),
                        user_id=message.from_user.id,
                        filename=document.file_name
                    )
                except:
                    await message.reply(f"🙁 Пожалуйста, отправьте файл `.zip` или текстовые файлы исходного кода (`.py`,`.js`,`.c` итд.)", parse_mode="markdown")
            else:
                downloaded_file = await self.bot.download_file(
                    file.file_path, f"./uploads/zip/{document.file_name}"
                )
                await self.file_process_reviewer.process_zip_file(
                    uploaded_zip_file=f"./uploads/zip/{document.file_name}",
                    file_name=document.file_name,
                    file_id=document.file_id,
                    user_id=message.from_user.id
                )
        except Exception as err:
            logger.error("File download failed", err)
        return

    async def handle_text_code(self, message: Message):
        """Handle direct code text messages"""
        # Validate if the message looks like code

        if not self._is_likely_code(message.text):
            await message.reply(text="🙁 Ах, похоже, вы отправили что-то, кроме кода....")
            return
        
        # Submit code for async processing
        await self.process_code_review(
            file_content=message.text,
            user_id=message.from_user.id,
            filename="code.py"
        )
        
        await message.reply("Наш шеф-повар читает ваш код. Скоро вы получите отзыв. 🗿")
        return

    def _is_likely_code(self, text: str = "") -> bool:
        """Basic heuristic to detect if text is likely code"""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'int', 'char', 'for', 'function ', 'const ', 'let ', 'return', 'break', 'while'
        ]
        return any(indicator in text for indicator in code_indicators)


    async def send_message_updates(self, user_id, message_text, parse_mode : str = "markdown"):
        await self.bot.send_message(chat_id=user_id, text=message_text, parse_mode=parse_mode)

    async def send_document_updates(self, user_id, document, caption : str = '😃 Вот ваш обзор кода 😃', parse_mode : str ="markdown"):
        await self.bot.send_document(chat_id=user_id, document=document, caption=caption, parse_mode=parse_mode)

    async def process_code_review(self, file_content: str, user_id: int, filename: str = None):
        """
        Async task for processing code review
        
        Args:
            file_content (str): Code content
            user_id (int): Telegram user ID
            filename (str): Name of the file
        """
        try:

            await self.code_reviewer.process_code_review(
               file_content=file_content, filename=filename, user_id=user_id
            )
            
            # Send review back to user
            await self.bot.send_message(
                chat_id=user_id,
                text=f"Мы едем быстро...🚗", parse_mode="markdown"
            )

       
        except Exception as e:
            logger.error(f"Code review error: {e}")
            await self.bot.send_message(
                chat_id=user_id, 
                text=f"😬 Кажется, произошла какая-то ошибка."
            )

    async def start(self):
        """Start the bot"""
        await self.dp.start_polling(self.bot)

async def main():
    # celery_app.start()
    bot = CodeReviewBot()
    await bot.start()

if __name__ == '__main__':
    asyncio.run(main())