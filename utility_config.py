import os
import dotenv
import getpass

dotenv.load_dotenv()


def get_cred():
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
