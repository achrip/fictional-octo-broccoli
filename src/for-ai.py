# %%
# %pip install -Uq \
# langchain langchain-community langchain-huggingface langchain-chroma \
# pypdf transformers accelerate Xformers InstructorEmbedding \
# sentencepiece bitsandbytes tiktoken chromadb typer semantic_split \
# cryptography

# %% [markdown]
# # Document Pre-Processing
# 
# Document pre-processing is split into two parts: 
# - Clean trailing/extra spaces
# - Splitting text into smaller chunks

# %%
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# %%
path = "../assets/ncvs_documents/"
loader = DirectoryLoader(path=path,
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
documents = loader.load()
len(documents)

# %% [markdown]
# ## Clean trailing or extra whitespaces.

# %%
import re

for i in range(len(documents)):
  cleaned_docs = re.sub("\s+", " ", documents[i].page_content)    # remove trailing spaces
  documents[i].page_content = cleaned_docs

# %% [markdown]
# ## Splitting into Chunks
# 
# We split the text into smaller chunks. For this experiment, we will stick to 512 `chunk_size` and 250 `chunk_overlap` to persist context between retrievals.
# This is also because we have to take into account the embedding model's maximum sequence length. 

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                          chunk_overlap=250,
                                          separators=["\n\n",
                                                      "\n",
                                                      " ",
                                                      ".",
                                                      ",",
                                                      "\u200b",  # Zero-width space
                                                      "\uff0c",  # Fullwidth comma
                                                      "\u3001",  # Ideographic comma
                                                      "\uff0e",  # Fullwidth full stop
                                                      "\u3002",  # Ideographic full stop
                                                      ""])

# %%
text = splitter.split_documents(documents)
print(f"Recursive: {len(text)}")

# %% [markdown]
# # ChromaDB Collections
# 
# Text chunks processed will be passed through an embedding model and saved into
# a ChromaDB database (collection).

# %% [markdown]
# we define a custom embedding function to use for embedding the texts before storing it into the database. 

# %%
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer

dir = "db"
client = chromadb.PersistentClient(path=dir)

class MyEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer('all-distilroberta-v1', device='cuda')
        # consider using voyage-law-2

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents
        sentences = input
        embeddings = self.model.encode(sentences)
        return embeddings.tolist()

embedding_function = MyEmbeddingFunction()


# %% [markdown]
# retrieve an existing collection from the database if it exsists. if not, we shall create a new collection
# and populate it with the documents.

# %%
try: 
    collection = client.get_collection(name="ncvs-idn", embedding_function=embedding_function)

except ValueError: 
    collection = client.create_collection(name="ncvs-idn", embedding_function=embedding_function)
    batches = create_batches(api=client, 
                            ids=["NCVS{n:03}".format(n=i) for i in range(1, len(text)+1)],
                            documents=[s.page_content for s in text], 
                            metadatas=[s.metadata for s in text])

    for batch in batches: 
        collection.upsert(
            ids=batch[0], 
            documents=batch[3], 
            metadatas=batch[2])

# %%
# If something went wrong, remove the collections
# client.delete_collection(name="ncvs-idn")

# %% [markdown]
# # Generation Model
# 
# In theory, we could use any generative models. But since there are hardware constraints, we opted to use OpenAI's models.

# %%
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_TEMPLATE = """
Use the following context (delimited by <ctx></ctx>), \
chat history (delimited by <hs></hs>) and source \
(delimted by <src></src>) to answer the question:
---------------------
{context}
---------------------
{history}
---------------------
{source}
---------------------
Question: {query}
"""

# %% [markdown]
# define a function to format the retrieved documents from the database

# %%
def generate_prompt_items(query: str): 
    retrieve = collection.query(
        query_texts=[query], 
        n_results=5
    )
    context = [["<ctx>" + s + "</ctx>" for s in chunk] for chunk in retrieve.get("documents")]
    context = "".join("\n\n".join(chunk) for chunk in context)

    source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieve.get("metadatas")]
    source  = "".join("\n\n".join(chunk) for chunk in source)
    return context, source
    

# %% [markdown]
# define a function to abstract the process of generating a response from the OpenAI model. 

# %%
def generate_openai_response(query): 
    context, source = generate_prompt_items(query)
    client = OpenAI(api_key=OPENAI_API_KEY)

    return client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages = [
            {
                "role":"system", 
                "content": "You will be provided a context (delimited by <ctx></ctx>) and the context source \
                            (delimited by <src></src>). Answer the question only based on the context given. \
                            Include the sources used in the answer you generated after the final paragraph, \
                            formatted with bullets for each different sources and sort it in a ascending manner."
            },
            {
                "role": "assistant",
                "content": context
            },
            {
                "role": "assistant",
                "content": source
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0
    ).choices[0].message.content