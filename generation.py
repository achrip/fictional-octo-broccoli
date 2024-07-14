from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from embeddings import DocumentRetriever
import os


class TextGenerationModel: 
    def __init__(self, k) -> None:
        self.k = k
        self.retriever = DocumentRetriever()
        pass

    def generate_prompt_items(self, query: str): 
        retrieve = self.retriever.collection.query(
                query_texts=[query], 
                n_results=self.k
                )

        context = [["<ctx>" + s + "</ctx>" for s in chunk] for chunk in retrieve.get("documents")]
        context = "".join("\n\n".join(chunk) for chunk in context)

        source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieve.get("metadatas")]
        source  = "".join("\n\n".join(chunk) for chunk in source)
        return context, source

    def generate_openai_response(self, query): 
        api_key = os.getenv('OPENAI_API_KEY') 
        context, source = self.generate_prompt_items(query)
        client = OpenAI(api_key=api_key)

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

    def generate_response(self, query): 
        # TODO: implement generation process for local LLMs
        pass

if __name__ == "__main__": 
    model = TextGenerationModel(7)
    model.__dict__
    context, source = model.generate_prompt_items("what should i consider when assembling the marine evacuation system?")
    print(f"context: {context}")
    print(f"sources: {source}")
