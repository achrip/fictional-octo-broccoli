import pandas as pd
from .generation import TextGenerationModel

data = {
        'Question': [
            ], 
        'Answer': [
            ]
        }

df = pd.DataFrame(data)

queries = df['Question']
targets = df['Answer']

def hitrate_at_k(query, source_targets): 
    retrieval_results = collection.query(
        query_texts=[query], 
        n_results=k)
    
    source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieval_results.get("metadatas")]

    count = 0
    for target in source_targets: 
        for retrieved in source[0]: 
            if target in retrieved.lower(): 
                count += 1
                continue
    
    return count/k

def recall_at_k(query, source_targets): 
    retrieval_results = collection.query(
        query_texts=[query], 
        n_results=k)
    
    source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieval_results.get("metadatas")]

    count = 0
    for target in source_targets: 
        for retrieved in source[0]: 
            if target in retrieved.lower(): 
                count += 1
                break
    
    return count/len(source_targets)

def precision_at_k(query, source_targets): 
    retrieval_results = collection.query(
        query_texts=[query], 
        n_results=k)
    
    source = [["<src>" + "Source: " + s["source"] + ", page: " + str(s["page"]) + "</src>" for s in chunk] for chunk in retrieval_results.get("metadatas")]

    count = 0
    for target in source_targets: 
        for retrieved in source[0]: 
            if target in retrieved.lower(): 
                count += 1
                continue
    
    return count/len(source[0])

def mean_reciprocal_rank(query, source_targets): 
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
    hit.append(hitrate_at_k(q, a))
    pre.append(precision_at_k(q, a))
    rec.append(recall_at_k(q, a))
    mrr.append(mean_reciprocal_rank(q, a))

print(f"Average hitrate over {k} queries: {sum(hit)/len(hit)}")
print(f"Average precision over {k} queries: {sum(pre)/len(pre)}")
print(f"Average recall over {k} queries: {sum(rec)/len(rec)}")
print(f"Average mean reciprocal ranks over {k} queries: {sum(mrr)/len(mrr)}")
