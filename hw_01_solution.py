"""from gitsource import GithubRepositoryDataReader

from minsearch import Index
from rag_helper import K2ThinkRAG

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os


from gitsource import chunk_documents


openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.k2think.ai/v1"
)

def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index

def search(query, index, num_results=1):

    return index.search(
        query,
        num_results=num_results,
    )



reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

chunks = chunk_documents(documents, size=2000, step=1000)
#print(len(chunks))

#print(doc)
#print(search("How does the agentic loop keep calling the model until it stops?", build_index(documents)))

assistant = K2ThinkRAG(build_index(chunks), openai_client)
print(assistant.rag("How does the agentic loop keep calling the model until it stops?"))"""


from gitsource import GithubRepositoryDataReader
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

from dotenv import load_dotenv
load_dotenv()
import json
import os
from minsearch import Index
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.k2think.ai/v1"
)

def build_index(documents):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"]
    )
    index.fit(documents)
    return index

def search(query, index, num_results=1):

    return index.search(
        query,
        num_results=num_results,
    )


instructions = """
You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering.
""".strip()

search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

def run_agent(question: str, max_iterations: int = 10):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question},
    ]
    search_calls = 0

    for _ in range(max_iterations):
        response = openai_client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=messages,
            tools=[search_tool],
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content, search_calls

        for tool_call in tool_calls:
            if tool_call.function.name != "search":
                continue

            search_calls += 1
            args = json.loads(tool_call.function.arguments)
            tool_result = search(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            })

    raise RuntimeError("Agent did not finish within max_iterations")

answer, search_calls = run_agent(
    "How does the agentic loop work, and how is it different from plain RAG?"
 )
print(answer)
print(search_calls)
