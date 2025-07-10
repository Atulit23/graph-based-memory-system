import time
from sklearn.metrics import f1_score
import numpy as np
from datasets import load_dataset
from superfinal import MemoryGraph
import os
import json

hotpot = load_dataset("hotpot_qa", "fullwiki", split="validation[:200]")

def evaluate_retrieval(graph, dataset, top_k=5):
    em_scores, f1_scores, latencies = [], [], []

    for item in dataset:
        question, gold = item["question"].lower(), item["answer"].lower()

        start = time.time()
        results = graph.query(question, top_k=top_k)
        latencies.append(time.time() - start)

        preds = [info["text"].split()[0].lower() for info in results.values()]

        em_scores.append(int(gold in preds))

        pred = max(preds, key=lambda x: f1_score(gold.split(), x.split()))
        f1_scores.append(f1_score(gold.split(), pred.split()))

    return {
        "EM": np.mean(em_scores),
        "F1": np.mean(f1_scores),
        "Latency": np.mean(latencies),
        "P95 Latency": np.percentile(latencies, 95)
    }


def load_chats_from_json(filepath, graph_path="graph16.json", limit=2563):
    graph = MemoryGraph()
    if os.path.exists(graph_path):
        graph.load_graph(graph_path)
        return graph

    with open(filepath, "r") as f:
        chats = json.load(f)

    for i, chat in enumerate(chats[:limit]):
        if i not in (355, 356, 357):
            print('----------' + str(i) + '/' + str(limit) + '----------')
            chat_text = chat.get("conversation", "")
            chat_id = chat.get("id")
            timestamp = chat.get("timestamp")
            graph.add_chat(chat_text, chat_id=chat_id, timestamp=timestamp)

    graph.build_edges()
    graph.save_graph(graph_path)
    return graph

graph = load_chats_from_json("combined_chats.json")
metrics = evaluate_retrieval(graph, hotpot, top_k=5)

print(metrics)