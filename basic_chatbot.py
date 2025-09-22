from utility_config import get_cred
import dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


get_cred()

dotenv.load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


workflow = StateGraph(state_schema=State)


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memoryview = MemorySaver()
app = workflow.compile(checkpointer=memoryview)

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
config = {"configurable": {"thread_id": "abc123"}}

query = "Hi I'm Todd, please tell me a joke."
language = "english"
input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")

output = app.invoke({"messages": input_messages, "language": "french"}, config)
output["messages"][-1].pretty_print()


query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": "french"}, config)
output["messages"][-1].pretty_print()
