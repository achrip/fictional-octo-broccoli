from generation import TextGenerationModel
from evaluation import *
import argparse 

parser = argparse.ArgumentParser()

if __name__ == "__main__": 
    system = TextGenerationModel(k=7)
    query = input("Ask me something about non-convention vessels regulations: ")
    response = system.generate_openai_response(query)
    print(response)
