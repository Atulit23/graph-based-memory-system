import json
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from datetime import datetime
import uuid
import os
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import Counter

class EmotionDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.labels = ['anger', 'joy', 'optimism', 'sadness']

    def detect(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0].numpy()
        return dict(zip(self.labels, probs.tolist()))

def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()
nlp = spacy.load("en_core_web_sm")

class MemoryGraph:
    def __init__(self, similarity_threshold=0.3):
        self.graph = nx.MultiGraph()
        self.similarity_threshold = similarity_threshold
        self.embeddings = []
        self.ed = EmotionDetector()
        self.global_keyword_freq = Counter()
        self.total_docs = 0

    def _generate_id(self):
        return str(uuid.uuid4())

    def _extract_entities_relations(self, text):
        doc = nlp(text)
        entities = [ent.text.lower() for ent in doc.ents 
                    if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"}]

        common_words = {
            "role", "level", "software", "engineer", "system", "work", "time", "way", "thing", "person", "place",
            "model", "chat", "user", "help", "question", "answer", "feature", "data", "information", "content",
            "text", "word", "language", "context", "token", "api", "service", "website", "app", "tool",
            "need", "want", "think", "know", "understand", "mean", "use", "make", "get", "give", "take",
            "good", "bad", "right", "wrong", "better", "best", "new", "old", "big", "small", "high", "low",
            "yeah", "okay", "sure", "thanks", "please", "hello", "bye", "yes", "no", "maybe", "actually",
            "basically", "essentially", "probably", "definitely", "really", "quite", "pretty", "very",
            "scrape", "extract", "generate", "create", "build", "search", "find", "look", "show", "tell",
            "explain", "describe", "discuss", "talk", "say", "ask", "wonder", "check", "try", "seem"
        }

        important_tokens = []
        for token in doc:
            if (token.pos_ in {"NOUN", "VERB", "ADJ"} and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 3 and
                token.text.lower() not in common_words and
                not token.text.lower().startswith(("http", "www")) and
                token.is_alpha):
                important_tokens.append(token.lemma_.lower())

        filtered_tokens = [t for t in important_tokens if t not in {"able", "available", "possible", "different", "important", "specific", "general", "particular"}]
        keywords = list(set(entities + filtered_tokens))
        keywords = [kw for kw in keywords if len(kw) >= 4]
        self.global_keyword_freq.update(keywords)
        return keywords

    def _calculate_keyword_importance(self, keywords):
        if self.total_docs == 0:
            return {kw: 1.0 for kw in keywords}

        importance = {}
        for kw in keywords:
            freq = self.global_keyword_freq.get(kw, 1)
            idf = np.log(self.total_docs / (freq + 1))
            if freq > self.total_docs * 0.1:
                idf *= 0.1
            elif freq > self.total_docs * 0.05:
                idf *= 0.3
            importance[kw] = max(0.01, idf)
        return importance

    def add_chat(self, chat_text, chat_id=None, timestamp=None, verification_score=1.0):
        chat_text = " ".join(chat_text.split()[:300])
        embedding = model.encode(chat_text, normalize_embeddings=True)
        embedding = embedding.astype(float).tolist()

        node_id = chat_id if chat_id else self._generate_id()
        keywords = self._extract_entities_relations(chat_text)
        emotions = self.ed.detect(chat_text)
        emotion_vec = list(emotions.values()) if emotions else [0.25] * 4

        metadata = {
            "text": chat_text,
            "embedding": embedding,
            "emotions": emotions,
            "emotion_vector": emotion_vec,
            "keywords": keywords
        }
        self.graph.add_node(node_id, **metadata)
        self.embeddings.append((node_id, np.array(embedding)))
        self.total_docs += 1

    def build_edges(self):
        print("Building semantic edges...")
        semantic_edges_created = 0
        for i, (id1, emb1) in tqdm(enumerate(self.embeddings), total=len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                id2, emb2 = self.embeddings[j]
                sim = np.dot(emb1, emb2)
                if sim >= self.similarity_threshold:
                    self.graph.add_edge(id1, id2, weight=sim, type="semantic")
                    semantic_edges_created += 1

        print(f"Created {semantic_edges_created} semantic edges")

        print("Building symbolic edges...")
        raw_symbolic_edges = []
        for i, (id1, _) in tqdm(enumerate(self.embeddings), total=len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                id2, _ = self.embeddings[j]

                kw1 = set(self.graph.nodes[id1]["keywords"])
                kw2 = set(self.graph.nodes[id2]["keywords"])
                shared = kw1 & kw2

                if len(shared) >= 2:
                    kw_importance = self._calculate_keyword_importance(list(shared))
                    importance_weight = sum(kw_importance.values())
                    avg_rarity = importance_weight / len(shared)

                    if avg_rarity < 0.3:
                        continue

                    raw_symbolic_edges.append({
                        "id1": id1,
                        "id2": id2,
                        "shared": list(shared),
                        "raw_weight": importance_weight
                    })

        symbolic_edges_created = 0
        if raw_symbolic_edges:
            weights = [e["raw_weight"] for e in raw_symbolic_edges]
            min_w = min(weights)
            max_w = max(weights)
            for e in raw_symbolic_edges:
                norm_weight = (e["raw_weight"] - min_w) / (max_w - min_w + 1e-6)
                self.graph.add_edge(
                    e["id1"],
                    e["id2"],
                    weight=norm_weight,
                    shared_keywords=e["shared"],
                    type="symbolic"
                )
                symbolic_edges_created += 1

        print(f"Created {symbolic_edges_created} symbolic edges")


    def query(self, input_text, top_k=5, alpha=0.5, beta=0.2, gamma=0.3):
      input_embedding = model.encode(input_text, normalize_embeddings=True).astype(np.float32)
      input_emotion = self.ed.detect(input_text)
      input_emotion_vector = np.array(list(input_emotion.values()))

      # Extract keywords from the query for symbolic matching
      input_keywords = set(self._extract_entities_relations(input_text))

      symbolic_boosts = {}
      max_boost = 0.0
      for node_id, data in self.graph.nodes(data=True):
          boost = 0.0
          for neighbor_id in self.graph.neighbors(node_id):
              for edge_data in self.graph.get_edge_data(node_id, neighbor_id).values():
                  if edge_data.get("type") == "symbolic":
                      shared_keywords = set(edge_data.get("shared_keywords", []))
                      if shared_keywords & input_keywords:
                          boost += edge_data.get("weight", 0.0)
          symbolic_boosts[node_id] = boost
          max_boost = max(max_boost, boost)

      if max_boost == 0:
          max_boost = 1e-6  # Avoid divide-by-zero

      scores = []
      for node_id, data in self.graph.nodes(data=True):
          sem_sim = np.dot(input_embedding, np.array(data["embedding"]))
          emo_vec = np.array(data.get("emotion_vector", [0.25]*4))
          emo_sim = np.dot(input_emotion_vector, emo_vec) / (
              np.linalg.norm(input_emotion_vector) * np.linalg.norm(emo_vec) + 1e-8
          )

          symbolic_boost = symbolic_boosts[node_id] / max_boost  # Normalize
          combined_score = alpha * sem_sim + beta * emo_sim + gamma * symbolic_boost

          scores.append((node_id, combined_score, sem_sim, emo_sim, symbolic_boost))

      scores.sort(key=lambda x: x[1], reverse=True)

      results = {}
      for rank, (node_id, score, sem_sim, emo_sim, symbolic_boost) in enumerate(scores[:top_k]):
          data = self.graph.nodes[node_id]
          results[node_id] = {
              "rank": rank + 1,
              "score": score,
              "semantic_sim": sem_sim,
              "emotion_sim": emo_sim,
              "symbolic_boost": symbolic_boost,
              "text": data["text"],
              "neighbors": []
          }

          neighbors = []
          for neighbor_id in self.graph.neighbors(node_id):
              neighbor = self.graph.nodes[neighbor_id]
              for edge_data in self.graph.get_edge_data(node_id, neighbor_id).values():
                  neighbors.append({
                      "id": neighbor_id,
                      "weight": edge_data.get("weight", 0),
                      "type": edge_data.get("type", "semantic"),
                      "shared_keywords": edge_data.get("shared_keywords", []),
                      "text": neighbor["text"][:300]
                  })

          neighbors.sort(key=lambda x: x["weight"], reverse=True)
          results[node_id]["neighbors"] = neighbors[:10]

      return results

    def save_graph(self, path="graph16.json"):
        data = nx.node_link_data(self.graph, edges="edges")
        metadata = {
            "global_keyword_freq": dict(self.global_keyword_freq),
            "total_docs": self.total_docs
        }
        cleaned_data = to_python_types(data)
        cleaned_data["metadata"] = metadata
        with open(path, "w") as f:
            json.dump(cleaned_data, f)

    def load_graph(self, path="graph16.json"):
        with open(path, "r") as f:
            data = json.load(f)
        if "metadata" in data:
            metadata = data.pop("metadata")
            self.global_keyword_freq = Counter(metadata.get("global_keyword_freq", {}))
            self.total_docs = metadata.get("total_docs", 0)
        self.graph = nx.node_link_graph(data, edges="edges")
        self.embeddings = [(n, np.array(d["embedding"], dtype=np.float32)) for n, d in self.graph.nodes(data=True)]

def load_chats_from_json(filepath, graph_path="graph16.json", limit=2563):
    graph = MemoryGraph()
    if os.path.exists(graph_path):
        graph.load_graph(graph_path)
        return graph

    with open(filepath, "r") as f:
        chats = json.load(f)

    for i, chat in enumerate(chats[:limit]):
        if i != 355 and i != 356 and i != 357:
            print('----------' + str(i) + '/' + str(limit) + '----------')
            chat_text = chat.get("conversation", "")
            chat_id = chat.get("id")
            timestamp = chat.get("timestamp")
            graph.add_chat(chat_text, chat_id=chat_id, timestamp=timestamp)

    graph.build_edges()
    graph.save_graph(graph_path)
    return graph

graph = load_chats_from_json("combined_chats.json")
results = graph.query("how do you think we can build jarvis & iron man's suit, who do you think will be able to do in real life")

for node_id, info in results.items():
    print(f"\nNode: {node_id}\nText: {info['text']}")
    for n in info['neighbors']:
        print(f"  ↳ Neighbor: {n['id']} | Text: {n['text']}  | Weight: {n['weight']:.2f}")