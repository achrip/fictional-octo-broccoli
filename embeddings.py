from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import re, chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer

class CustomEmbeddingFunction(EmbeddingFunction[Documents]): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer('all-distilroberta-v1', device='cuda')

    def __call__(self, input: Documents) -> Embeddings:
        sentences = input
        embeddings = self.model.encode(sentences)
        return embeddings.tolist()

class DocumentRetriever: 
    def __init__(self):
        self.__dir = 'db'
        self.__client = chromadb.PersistentClient(path=self.__dir)
        self.__embedding_function = CustomEmbeddingFunction()

        documents = self._get_documents()
        documents = self._clean_documents(documents)
        raw_text = self._split_documents(documents, 512, 250) # make sure chunk size and overlap
                                                              # matches the embedding model limit.

        try: 
            self.collection = self.__client.get_collection(name='statutoryDB', 
                                                         embedding_function=self.__embedding_function)
        except ValueError: 
            self.collection = self.__client.create_collection(name='statutoryDB', 
                                                            embedding_function=self.__embedding_function)
            self.__populate_database(raw_text)


    def _get_documents(self): 
        path = './assets/'
        loader = DirectoryLoader(path=path, 
                                 glob='*/*.pdf', 
                                 loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents

    def _clean_documents(self, documents): 
        for i in range(len(documents)): 
            cleaned_texts = re.sub('\s+', ' ', documents[i].page_content)
            documents[i].page_content = cleaned_texts

        return documents

    def _split_documents(self, 
                        documents,
                        size: int, 
                        overlap: int):
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, 
                                                  chunk_overlap=overlap, 
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
        raw_text = splitter.split_documents(documents)
        return raw_text

    def __populate_database(self, raw_text): 
        batches = create_batches(api=self.client, 
                                 ids=["NCVS{n:03}".format(n=i) for i in range(1, len(raw_text)+1)],
                                 documents=[s.page_content for s in raw_text], 
                                 metadatas=[s.metadata for s in raw_text])
        for batch in batches: 
            self.collection.upsert(
                    ids=batch[0], 
                    documents=batch[3], 
                    metadatas=batch[2])
