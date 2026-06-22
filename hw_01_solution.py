from gitsource import GithubRepositoryDataReader

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
print(assistant.rag("How does the agentic loop keep calling the model until it stops?"))


