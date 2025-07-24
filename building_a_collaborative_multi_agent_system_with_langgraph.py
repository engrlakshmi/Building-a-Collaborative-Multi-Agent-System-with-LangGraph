
!pip install langchain



!pip install langchain-community

"""The error `ModuleNotFoundError: No module named 'langchain_community'` indicates that the necessary package containing the `HuggingFaceHub` class was not found in your environment. Installing `langchain-community` should resolve this issue."""

from langchain.llms import HuggingFaceHub

"""Hugging face google flan t5

To use the Hugging Face Hub models, you need an API token. You can get one from your Hugging Face account settings. Once you have the token, you can add it to Colab's secrets manager by clicking on the "🔑" icon in the left sidebar. Name the secret `HUGGINGFACEHUB_API_TOKEN`.

Then, you can access the token in your code like this:
"""

import os
from google.colab import userdata
HF_TOKEN ='xx'

os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get('HF_TOKEN')

"""Using Local Transformers"""

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

"""Each LLM can be wrapped as a function or LangChain Runnable and used as a node in the graph.


"""

!pip install langgraph

from langgraph.graph import END, StateGraph
from typing import TypedDict

# ---- Define State ----
class AgentState(TypedDict):
    input: str
    summary: str
    answer: str
    chat_response: str

# ---- Load Hugging Face Pipelines ----
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
summarizer =pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    max_length=60,
    min_length=20,
    do_sample=False
)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
chat_model = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=200
)

# ---- Wrap with LangChain ----
summarizer_llm = HuggingFacePipeline(pipeline=summarizer)
chat_llm = HuggingFacePipeline(pipeline=chat_model)

# ---- Agent Functions ----

def summarize_node(state: AgentState) -> AgentState:
    text = state['input']
    summary = summarizer_llm.invoke(text)
    return {**state, "summary": summary}

def qa_node(state: AgentState) -> AgentState:
    context = state.get("summary") or state["input"]
    question = "What is the main topic?"
    result = qa_pipeline(question=question, context=context)
    return {**state, "answer": result["answer"]}

def chat_node(state: AgentState) -> AgentState:
    input_text = state['input']
    chat_output = chat_llm.invoke(input_text)
    return {**state, "chat_response": chat_output}

# ---- Build Graph ----
graph_builder = StateGraph(AgentState)

graph_builder.add_node("summarizer", summarize_node)
graph_builder.add_node("qa", qa_node)
graph_builder.add_node("chat", chat_node)

# Flow: summarize → qa → chat → END
graph_builder.set_entry_point("summarizer")
graph_builder.add_edge("summarizer", "qa")
graph_builder.add_edge("qa", "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()

# ---- Run the Graph ----
if __name__ == "__main__":
    user_input = "LangGraph is a library built on top of LangChain that enables creating graphs of language model agents to solve complex workflows collaboratively."
    result = graph.invoke({"input": user_input})

    print("\n--- Final Output ---")
    print(f"Summary: {result['summary']}")
    print(f"Answer: {result['answer']}")
    print(f"Chat: {result['chat_response']}")
