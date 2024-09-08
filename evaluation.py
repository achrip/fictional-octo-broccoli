import pandas as pd
from generation import TextGenerationModel

data = {
        'Question': [
            ], 
        'Answer': [
            ]
        }

df = pd.DataFrame(data)

queries = df['Question']
targets = df['Answer']


def mean_reciprocal_rank(query,
                         source_targets, 
                         k, 
                         collection): 
    retrieval_results = collection.query(
        query_texts=[query], 
        n_results=k)
    
    source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieval_results.get("metadatas")]

    reciprocal_ranks = [1/(i+1) for target in source_targets for i, result in enumerate(source[0]) if target in result.lower()]
    try: 
        return sum(reciprocal_ranks)/len(reciprocal_ranks)
    except ZeroDivisionError: 
        return 0

# Average score of each metrics

# Note: Shouldn't be average, but a graph showing which k value is optimal.
# TODO: create new logic

hit = []
pre = [] 
rec = []
mrr = []

for i, (q, a)in enumerate(zip(queries, targets)): 
    mrr.append(mean_reciprocal_rank(q, a))

print(f"Average mean reciprocal ranks over {k} queries: {sum(mrr)/len(mrr)}")
