import logging
from typing import Dict


# Import necessary local modules
from src.analyzer import CodeAnalyzer
import os, zipfile
from src.gen_report import MarkdownToPDFConverter
from dotenv import load_dotenv
from aiogram.types import FSInputFile

load_dotenv(".env")

logger = logging.getLogger(__name__)


class CodeReviewWorker:
    def __init__(
        self, 
        send_message_updates,
        send_document_updates,
        llm_api_key : str,
        llm_base_url : str,
        vector_store_path : str = "./faiss-store/",
        model_name : str = "mistral-nemo-instruct-2407",
    ):
        """
        Initialize workers with necessary services
        
        Args:
            send_message_updates (method): Function to send message to user
            send_document_updates (method): Function to send file to user
            llm_api_key (str): API key for LLM service
            llm_base_url (str): LLM model connection URL
            vector_store_path (str): Vector store data folder
            model_name (str): Name of the LLM model
            
        """
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.send_message_updates = send_message_updates
        self.send_document_updates = send_document_updates

        self.code_analyzer = CodeAnalyzer(
            llm_api_key = self.llm_api_key,
            llm_base_url = self.llm_base_url,
            guidelines = self._load_guidelines(),
            vector_store_path = self.vector_store_path,
            model_name=model_name,
            send_message_updates=self.send_message_updates
        )
        self.vector_store_path = vector_store_path

    def _load_guidelines(self) -> Dict[str, str]:
        """
        Load code guidelines from vector database
        
        Returns:
            Dict of code guidelines 
        """
        try:
            guidelines =  {
                "Code Organization" : "The code should be structured according to monorepo principles, using .gitignore and .editorconfig files.",
                "Using Standard Tools and Libraries" : "Using popular and supported(latest) versions of packages such as Falcon, Gunicorn, Gevent, Alembic and others.",
                "Verification and Testing" : "Availability of tests, both unit tests and integration tests.",
                "Logging and Monitoring" : "Correct use of the logging module and log recording formats.",
                "Authentication and Authorization" : "Checking the correct processing of JWT tokens and protection of access to resources.",
                "Compliance with Stylistic Standards" : "Adherence to PEP8 code formatting rules and availability of PEP256 and PEP257 documentation.",
                "Performance and Optimizattion" : "It is also worth paying attention to the use of dialect-dependent constructs and performance optimization, especially when working with large amounts of data.",
            }
            return guidelines
        except Exception as e:
            logger.error(f"Failed to load guidelines: {e}")
            return {}

    async def process_code_review(
        self,
        file_content: str, 
        user_id: int,
        filename: str = "code.py",
        single_file : bool = True
    ) :
        """
        Asynchronous task for processing code review
        
        Args:
            file_content (str): Code to analyze
            user_id (int): Telegram user ID
            filename (str): Name of the file
            single_file (bool): if user uploaded a single file or a zip file
        
        Returns:
            Dict with code review results
        """
        
        try:
            # Initialize workers with environment configurations
            # Perform code analysis
            file_content = file_content.replace("\n\n", "")

            analysis_result = self.code_analyzer.analyze(file_content, file_name=filename)
            
            # Log analysis result
            logger.info(f"Completed code review for {filename}")

            if single_file:

                await self.send_message_updates(user_id=user_id, message_text="Обзор кода был сгенерирован ✅", parse_mode="markdown")
                
                converter = MarkdownToPDFConverter(
                    output_file=f"./pdfs/{filename.split(".")[0]}"
                )
                converter.convert(markdown_text=analysis_result)
                
                await self.send_message_updates(user_id=user_id, message_text="**🔥🔥 Ваш PDF-файл обзора кода готов.**", parse_mode="markdown")
                
                await self.send_document_updates(user_id=user_id, document= FSInputFile(path=f"./pdfs/{filename.split(".")[0]}.pdf", filename=filename.split(".")[0]+".pdf"))

            else:
                return analysis_result
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")


class FileProcessReviewWorker:
    def __init__(
        self,
        send_message_updates,
        send_document_updates,
        code_reviewer : CodeReviewWorker
    ):
        self.code_reviewer = code_reviewer
        self.send_message_updates = send_message_updates
        self.send_document_updates = send_document_updates

    def _extract_zip_file(
        self,
        uploaded_zip_file : str,
        file_name : str,
        file_id : str = None
    ):
        
        try:
            extract_path = os.path.join("./uploads/extract/", file_id) #  os.path.splitext(file_name)[0])
            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            return extract_path
        except zipfile.BadZipFile:
                logger.info("The uploaded file is not a valid ZIP file.")
                return False
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return False

    # @celery_app.task(bind=True, max_retries=3)
    async def process_zip_file(
        self,
        user_id : int,
        uploaded_zip_file : str,
        file_name : str,
        file_id : str = None,
    ):
        await self.send_message_updates(user_id=user_id, message_text="**📂 Извлечение .zip-файлаe**", parse_mode="markdown")
        extract_path = self._extract_zip_file(
            uploaded_zip_file=uploaded_zip_file, file_name=file_name, file_id=file_id
        )

        if extract_path:
            project_report = ""
            await self.send_message_updates(user_id=user_id, message_text="Готовится обзор кода...", parse_mode="markdown")

            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                            project_report += await self.code_reviewer.process_code_review(
                                file_content=file_content,
                                filename=file_path,
                                single_file=False,
                                user_id=user_id
                            )
                            f.close()

                        f.close()
                    except Exception as err:
                        logger.error(err)
            await self.send_message_updates(user_id=user_id, message_text="Обзор кода был сгенерирован ✅", parse_mode="markdown")

            converter = MarkdownToPDFConverter(
                output_file=f"./pdfs/{file_id}"
            )
            converter.convert(markdown_text=project_report)
            await self.send_message_updates(user_id=user_id, message_text="**🔥🔥 Ваш PDF-файл обзора кода готов.**", parse_mode="markdown")

            await self.send_document_updates(user_id=user_id, document=FSInputFile(path=f"./pdfs/{file_id}.pdf", filename=file_name+".pdf"))

            try:
                os.rmdir(extract_path)
            except:
                pass
        else:
            logger.error("ZIP FILE EXTRACTION FAILED")