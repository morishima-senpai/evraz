from langchain.prompts import PromptTemplate
import subprocess
from typing import Dict, List, Any, Optional

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import logging, os
import tempfile, requests

from celery import Celery
from dotenv import load_dotenv
load_dotenv(".env")


logger = logging.getLogger(__name__)


celery_app = Celery(
    'evraz_code_review_bot', 
    broker = os.environ.get('REDIS_URL'), 
    backend = os.environ.get('REDIS_URL')
)
celery_app.autodiscover_tasks()



class LLMAPI:
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None, guidelines : Dict = {}, model_name : str = "mistral-nemo-instruct-2407" ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        self.guidelines = "\n\n".join(f"{key} : {value}" for key, value in guidelines.items())

    def answer(self, prompt: str, stop: Optional[List[str]] = None, history : List = []) -> str:
        messages = [{"role" : "system", "content" : f"""Analyze this code and the provided analysis report against our company guidelines and best practices and Generate a detailed code review report. Give a very little description. Be to the point. Summarize and generate a professional Code Review Report with Code Blocks and diff. Write very small description only if needed (1 or 2 lines). I am asenior engineer. And remember to answer in RUSSIAN. and DO NOT ANSWER AS A PARAGRAH, MAINTAIN BEUTIFUL MARKDOWN FORMAT AND Totally FINISH THE ANSWER.
GUIDELINES:                  
{self.guidelines}


Please provide a detailed review covering:

1. Performance optimizations
2. Security vulnerabilities
3. Code style and maintainability
4. Potential refactoring suggestions
5. Best Practices and Other important things

ALWAYS REPLY IN RUSSIAN LANGUAGE
"""
            },
            {"role" : "user", "content" : """This is a tempalte or format for how to reply. Try to reply in like this format :
Анализ проекта ```project_name``` от 01.01.2024 00:00:00 UTC+3
         
Дата последнего изменения проекта : 01.01.2024 00:00:00 UTC+3

Общее количество ошибок: 3
Архитектурных нарушений: 1
... _<Перечисление других классов ошибок>_
Несоответствий стандартам: 1 

### Архитектурное нарушение
> `chat_service.py` (номер строки:номер символа, при наличии)
>  Необходимо вынести в слой адаптеров, работать через репозитории и интерфейсы из сервисов

```python
user = User.query.filter_by(username=token).first()
location = Location.query.filter_by(name=name).first()
```

### Краткое описание нарушения (Add braces to if statement)
> `LinkFragmentValidator.cs` (номер строки:номер символа, при наличии)
> `Severity`	`Code`	`Description`	`Project`	`File`	`Line`	
> Error (active)	RCS1007	Add braces to if statement	Eurofurence.App.Server.Services	LinkFragmentValidator.cs    35

```csharp
if (!Guid.TryParse(fragment.Target, out Guid dealerId))
    return ValidationResult.Error("Target must be of typ Guid");
```
> Предложенное исправление

```csharp
if (!Guid.TryParse(fragment.Target, out Guid dealerId)) {
    return ValidationResult.Error("Target must be of typ Guid");
}
```

### Некорректное наименование
> `ui.tsx` (номер строки:номер символа, при наличии)
> Поскольку этот тип относится к компоненту ProductItem и отражает его интерфейс, то тип должен называться ProductItemProps

```ts
type ProductProps = {
  product: Product;
  theme: Theme;
  setProduct: (product: Product) => void;
};
```

> Предложенное исправление

```ts
type ProductItemProps = {
  product: Product;
  theme: Theme;
  setProduct: (product: Product) => void;
};
```"""},
            {"role" : "assistant", "content" : "Okay, I will create a Professional Code Review Report in Russian like given in the following format"},
            {"role" : "user", "content" : prompt}
        ]
        payload = {
            "model" : self.model_name,
            "messages" :  messages,
            "max_tokens": 1024,
            "temperature" : 0.1
        }

        try:
            # Send request to model endpoint
            response = requests.post(
                self.endpoint_url, 
                json=payload,
                headers={'Content-Type': 'application/json', "Authorization": self.api_key if self.model_name == "mistral-nemo-instruct-2407" else f'Bearer {self.api_key}'}
            )
                
            # Check response
            if response.status_code == 200:
                result = response.json()
                # Extract the model's response text
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Received status code {response.status_code}. Response: {response.text}"

        except Exception as e:
            return e

class CodeAnalyzer:
    def __init__(
        self,
        send_message_updates,
        llm_api_key: str,
        llm_base_url : str,
        model_name : str,
        guidelines : Dict[str, str] = None,
        vector_store_path : str = "./faiss-store/",
        embedding_model='microsoft/codebert-base-mlm',
    ):
        """
        Initialize Code Analyzer with LLM and custom guidelines
        
        Args:
            send_message_updates (method) : Function to send message to user
            llm_api_key (str): API key for language model
            llm_base_url (str): LLM model URL endpoint
            model_name (str) : Name of the LLM model
            guidelines (dict): Specific code quality guidelines
            vector_store_path (str) : Vector store folder
            embedding_model (str) : Name of the model used to create the embeddings
        """
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.guidelines = guidelines
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.guidelines = guidelines
        self.embedding_model = embedding_model
        # index = FAISS.
        self.llm_chat_api = LLMAPI(endpoint_url=self.llm_base_url, api_key=self.llm_api_key, guidelines=self.guidelines, model_name=self.model_name)

        # Load embeddings model
        self.embedding_model_hf = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        self.vectorstore = FAISS.load_local(self.vector_store_path, self.embedding_model_hf, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever()

    def _generate_pylint(self, file_content : str, file_name : str = "code.py"):
        """Runs pylint on a Python file."""
        try:

            if file_name != "code.py" and not file_name.endswith(".py"):
                return ""
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, delete_on_close=True) as tmp_file:
                tmp_file.write(file_content.encode("utf-8"))
                tmp_file_name = tmp_file.name
                result = subprocess.run(["pylint", tmp_file_name], capture_output=True, text=True)
                result = result.stdout.replace(tmp_file_name, file_name)
                tmp_file.close()
            
            tmp_file.close()
            return result
        except Exception as err:
            logger.error(err)
            return ""
        
    def _static_code_analysis(self, file_content: str, file_name : str = "code.py") -> Dict[str, Any]:
        """Perform AST-based static code analysis"""
        try:
            # tree = ast.parse(file_content)
            # complexity = ast.NodeVisitor()
            # Custom complexity calculation

            # tree = ast.parse(file_content)
            # visitor = ComplexityVisitor()
            # visitor.visit(tree)

            
            return {
                # 'Complexity': complexity.generic_visit(tree),
                'Ast Issues': "", # ,[], # self._detect_ast_issues(tree),
                'Pylinit analysis' : self._generate_pylint(file_content=file_content, file_name=file_name)
            }
        except SyntaxError as e:
            logger.error(f"Static Code Analysis Failed: {str(e)}")
            return {}
    
    def _security_scan(self, file_content: str, file_name : str = "code.py") -> Dict[str, Any]:
        """
        Security scanning using Bandit
        Detects potential security vulnerabilities
        """

        try:
            if file_name != "code.py" and not file_name.endswith(".py"):
                return ""
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, delete_on_close=True) as tmp_file:

                tmp_file.write(file_content.encode("utf-8"))
                tmp_file_name = tmp_file.name
                result = subprocess.run(["bandit", tmp_file_name], capture_output=True, text=True)
                result = result.stdout.replace(tmp_file_name, file_name)
                tmp_file.close()
            
            tmp_file.close()
            return result
        except Exception as err:
            logger.error(err)
            return ""
        
    def _performance_metrics(self, file_content: str, file_name : str = "code.py") -> Dict[str, float]:
        """
        Calculate Radon performance metrics
        
        Metrics:
        - Cyclomatic Complexity
        - Maintainability Index
        - Lines of Code
        """
        try:
            if file_name != "code.py" and not file_name.endswith(".py"):
                return {}
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, delete_on_close=True) as tmp_file:
                tmp_file_name = tmp_file.name
                result_c = subprocess.run(["radon", "cc", "-a", tmp_file_name], capture_output=True, text=True)
                result_m = subprocess.run(["radon", "mi", "-s" , "-x", tmp_file_name], capture_output=True, text=True)

                result_c = result_c.stdout.replace(tmp_file_name, file_name)
                result_m = result_m.stdout.replace(tmp_file_name, file_name)
                tmp_file.close()
            tmp_file.close()


            return {
                'Cyclomatic Complexity': result_c,
                'Maintainability': result_m,
                'Line of code': len(file_content.splitlines())
            }
        except Exception as err:
            logger.error(err)
            return {}
        


    def _local_analysis(self, file_content : str, file_name : str = "code.py") -> Dict[Any, Any]:
        try:
            analysis = {
                'Static Analysis': "\n\n".join(
                    f"{key}:\n {value}" 
                    for key, value in 
                    self._static_code_analysis(file_content, file_name=file_name).items()
                ) ,
                'Security Scan': self._security_scan(file_content, file_name=file_name),
                'Permance Metrics':"\n\n".join(
                    f"{key}: {value}" 
                    for key, value in 
                    self._performance_metrics(file_content, file_name=file_name).items()
                ),
                # 'llm_review': self._llm_code_review(code)
            }

            return analysis
        except Exception as err:
            logger.err(err)
            return {}
        

    def _llm_code_review(self, analysis : Any, file_content: str) -> str:
        """
        Use LLM for intelligent code review
        
        Incorporates:
        - Custom guidelines
        - General best practices
        - Specific language idioms
        """

        try:
            template = """Analysis results:
{source_analysis}
            
Code to Review:
```
{code}
```
"""
            prompt = PromptTemplate(
                input_variables=["source_analysis", "code"],
                template=template
            )

            context = ""
            try:
                context = self._retrieve_context(question="\n".join(f"{key} : {value}" for key, value in self.guidelines.items()))
            except Exception as e:
                logger.error(f"Context Retrival Failed: {str(e)}")
                return f"Context Retrival Failed: {str(e)}"
            
            prompt = prompt.format(
                source_analysis = "\n\n".join(f"{key}:\n {value}" for key, value in analysis.items()),
                code = file_content
            )

            messages = []
            first_part = self.llm_chat_api.answer(prompt=prompt, history=messages)
            # messages.extend([
            #     {"role" : "user", "content" : prompt},
            #     {"role" : "assistant", "content" : first_part}
            # ])
            # next_prompt = f'Ok Summarize and generate a professional Code Review Report with Code Blocks and diff. Write very small description only if needed (1 or 2 lines). I am asenior engineer. And remember to answer in RUSSIAN. and DO NOT ANSWER AS A PARAGRAH, MAINTAIN BEUTIFUL MARKDOWN FORMAT AND Totally FINISH THE ANSWER. \nhere is code again :\n ```\n{file_content}\n```'
            
            # second_part = self.llm_chat_api.answer(prompt=next_prompt, history=messages)
            # messages.extend([
            #     {"role" : "user", "content" : next_prompt},
            #     {"role" : "assistant", "content" : second_part}
            # ])      


            return first_part

        except Exception as err:
            logger.error("LLM review failed", err)
            return "LLM review failed (REPORT)"
    
    def _retrieve_context(self, question, k=3):
        """
        Retrieve relevant context from vector store
        
        Args:
            question (str): Input question
            k (int): Number of top context pieces to retrieve
        
        Returns:
            str: Combined context from retrieved documents
        """
        # Use vector store retriever to find relevant documents

        try:
            docs = self.vectorstore.similarity_search(question)        
            contexts = []
            for doc in docs:
                contexts.append(f"File: {doc.metadata['source']}\nConteqnt: {doc.page_content}")
            
            return "\n\n---\n\n".join(contexts)
        
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""

    def analyze(self, file_content: str, file_name : str = "code.py") -> str:
        """
        Comprehensive code analysis
        
        Returns:
            str : review report
        """
        try:
            analysis = self._local_analysis(file_content=file_content, file_name=file_name)
            return self._llm_code_review(analysis=analysis, file_content=file_content)
        except Exception as err:
            logger.error(err)
            return ""