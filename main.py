import getpass
import os
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

dotenv.load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = dotenv.get_key(
        dotenv.find_dotenv(), "GOOGLE_API_KEY"
    )

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
if not os.environ.get("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = "default"


model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

system_template = "Translate the follwoing from English into {language}."

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke(
    {
        "language": "Spanish",
        "text": "Let's play football!",
    }
)

# prompt.to_messages()

response = model.invoke(prompt)
print(response.content)
