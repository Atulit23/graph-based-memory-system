import random
import time
from superfinal import MemoryGraph
import numpy as np
import os
import json
import tracemalloc
from collections import Counter
import math

def analyze_result_diversity(results):
    texts = [info["text"].lower() for info in results.values()]
    words = [word for text in texts for word in text.split()]
    unique_words = set(words)
    
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    
    entropy = -sum((count / total_words) * math.log2(count / total_words)
                   for count in word_counts.values() if count > 0)
    
    avg_length = np.mean([len(text.split()) for text in texts])
    uniqueness_ratio = len(unique_words) / total_words if total_words else 0
    
    return {
        "unique_words": len(unique_words),
        "total_words": total_words,
        "entropy": entropy,
        "avg_length": avg_length,
        "uniqueness_ratio": uniqueness_ratio
    }

def quick_benchmark(graph):
    """Quick and simple benchmark you can run immediately"""

    tracemalloc.start()
    mem_before, _ = tracemalloc.get_traced_memory()
    
    # 1. Basic Stats
    print(f"Graph Stats:")
    print(f"  Total Nodes: {graph.graph.number_of_nodes()}")
    print(f"  Total Edges: {graph.graph.number_of_edges()}")
    print(f"  Average Degree: {np.mean([d for n, d in graph.graph.degree()]):.2f}")
    
    print(f"\nTesting Query Performance...")
    
    node_texts = [data["text"][:50] for _, data in graph.graph.nodes(data=True)]
    test_queries = random.sample(node_texts, min(20, len(node_texts)))
    
    query_times = []
    for i, query in enumerate(test_queries):
        start_time = time.time()
        results = graph.query(query, top_k=5)
        end_time = time.time()
        query_times.append(end_time - start_time)
        print(f"  Query {i+1}/20: {end_time - start_time:.4f}s", end="\r")
    
    print(f"\nQuery Performance Results:")
    print(f"  Average Query Time: {np.mean(query_times):.4f}s")
    print(f"  Fastest Query: {np.min(query_times):.4f}s")
    print(f"  Slowest Query: {np.max(query_times):.4f}s")
    print(f"  Queries per Second: {1/np.mean(query_times):.2f}")
    
    print(f"\nQuery Diversity / Uniqueness Check:")
    diversity_scores = []
    for query in test_queries[:5]:
        results = graph.query(query, top_k=5)
        metrics = analyze_result_diversity(results)
        diversity_scores.append(metrics["uniqueness_ratio"])
        print(f"  Query: '{query[:30]}...': Uniqueness = {metrics['uniqueness_ratio']:.3f}, Entropy = {metrics['entropy']:.2f}")

    print(f"  Avg Uniqueness Ratio: {np.mean(diversity_scores):.3f}")
    
    print(f"\nTesting Query Relevance...")
    
    relevance_tests = [
        {
            "query": "machine learning and artificial intelligence",
            "expected_keywords": ["machine", "learning", "artificial", "intelligence", "model", "algorithm", "neural", "network"]
        },
        {
            "query": "programming and software development",
            "expected_keywords": ["programming", "software", "development", "code", "python", "javascript", "developer"]
        },
        {
            "query": "personal relationships and emotions",
            "expected_keywords": ["relationship", "emotion", "personal", "feeling", "family", "friend", "love"]
        }
    ]
    
    relevance_scores = []
    for test in relevance_tests:
        results = graph.query(test["query"], top_k=5)
        
        found_keywords = set()
        for node_id, info in results.items():
            node_keywords = graph.graph.nodes[node_id].get("keywords", [])
            found_keywords.update([kw.lower() for kw in node_keywords])
            text_words = info["text"].lower().split()
            found_keywords.update(text_words)
        
        expected_set = set([kw.lower() for kw in test["expected_keywords"]])
        intersection = expected_set & found_keywords
        
        relevance = len(intersection) / len(expected_set) if expected_set else 0
        relevance_scores.append(relevance)
        
        print(f"  Query: '{test['query'][:40]}...'")
        print(f"    Relevance Score: {relevance:.2f}")
        print(f"    Found Keywords: {len(intersection)}/{len(expected_set)}")
    
    print(f"\nAverage Relevance Score: {np.mean(relevance_scores):.2f}")
    
    print(f"\nEdge Analysis:")
    semantic_edges = sum(1 for _, _, data in graph.graph.edges(data=True) if data.get("type") == "semantic")
    symbolic_edges = sum(1 for _, _, data in graph.graph.edges(data=True) if data.get("type") == "symbolic")
    
    print(f"  Semantic Edges: {semantic_edges}")
    print(f"  Symbolic Edges: {symbolic_edges}")
    print(f"  Edge Ratio (Semantic/Symbolic): {semantic_edges/symbolic_edges:.2f}" if symbolic_edges > 0 else "  No symbolic edges found")
    
    print(f"\nParameter Sensitivity Test:")
    test_query = "artificial intelligence machine learning"
    
    param_tests = [
        {"alpha": 0.9, "beta": 0.05, "gamma": 0.05, "name": "Semantic Heavy"},
        {"alpha": 0.33, "beta": 0.33, "gamma": 0.34, "name": "Balanced"},
        {"alpha": 0.2, "beta": 0.6, "gamma": 0.2, "name": "Emotion Heavy"},
        {"alpha": 0.2, "beta": 0.2, "gamma": 0.6, "name": "Symbolic Heavy"}
    ]
    
    for params in param_tests:
        results = graph.query(test_query, top_k=5, 
                             alpha=params["alpha"], 
                             beta=params["beta"], 
                             gamma=params["gamma"])
        avg_score = np.mean([info["score"] for info in results.values()])
        print(f"  {params['name']}: Average Score = {avg_score:.4f}")
    
    mem_after, _ = tracemalloc.get_traced_memory()
    print(f"\nMemory Usage:")
    print(f"  Before Benchmark: {mem_before / 1e6:.2f} MB")
    print(f"  After Benchmark:  {mem_after / 1e6:.2f} MB")
    tracemalloc.stop()
    
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

def detailed_query_analysis(graph, query, top_k=5):
    """Detailed analysis of a specific query"""
    
    print(f"\nDETAILED QUERY ANALYSIS")
    print(f"Query: '{query}'")
    print("-" * 50)
    
    start_time = time.time()
    results = graph.query(query, top_k=top_k)
    query_time = time.time() - start_time
    
    print(f"Query Time: {query_time:.4f}s")
    print(f"Results Found: {len(results)}\n")
    
    for i, (node_id, info) in enumerate(results.items(), 1):
        print(f"Result {i}:")
        print(f"  Node ID: {node_id}")
        print(f"  Overall Score: {info['score']:.4f}")
        print(f"  Semantic Similarity: {info['semantic_sim']:.4f}")
        print(f"  Emotion Similarity: {info['emotion_sim']:.4f}")
        print(f"  Symbolic Boost: {info['symbolic_boost']:.4f}")
        print(f"  Text Preview: {info['text'][:100]}...")
        print(f"  Keywords: {graph.graph.nodes[node_id].get('keywords', [])[:5]}")
        print(f"  Neighbors: {len(info['neighbors'])}\n")

def compare_queries(graph, queries):
    """Compare multiple queries side by side"""
    
    print(f"\nQUERY COMPARISON")
    print("-" * 50)
    
    results_comparison = {}
    
    for query in queries:
        start_time = time.time()
        results = graph.query(query, top_k=5)
        query_time = time.time() - start_time
        
        avg_score = np.mean([info["score"] for info in results.values()])
        avg_semantic = np.mean([info["semantic_sim"] for info in results.values()])
        avg_emotion = np.mean([info["emotion_sim"] for info in results.values()])
        avg_symbolic = np.mean([info["symbolic_boost"] for info in results.values()])
        
        results_comparison[query] = {
            "avg_score": avg_score,
            "avg_semantic": avg_semantic,
            "avg_emotion": avg_emotion,
            "avg_symbolic": avg_symbolic,
            "query_time": query_time,
            "num_results": len(results)
        }
    
    print(f"{'Query':<30} {'Avg Score':<10} {'Semantic':<10} {'Emotion':<10} {'Symbolic':<10} {'Time':<8}")
    print("-" * 80)
    
    for query, metrics in results_comparison.items():
        query_short = query[:25] + "..." if len(query) > 25 else query
        print(f"{query_short:<30} {metrics['avg_score']:<10.4f} {metrics['avg_semantic']:<10.4f} "
              f"{metrics['avg_emotion']:<10.4f} {metrics['avg_symbolic']:<10.4f} {metrics['query_time']:<8.4f}")

def load_chats_from_json(filepath, graph_path="graph20.json", limit=2563):
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

if __name__ == "__main__":
    graph = load_chats_from_json("combined_chats.json")
    quick_benchmark(graph)
    
    detailed_query_analysis(graph, "how do you think we can build jarvis & iron man's suit")
    
    test_queries = [
        "artificial intelligence and machine learning",
        "programming and software development", 
        "personal relationships and emotions",
        "creative writing and storytelling"
    ]
    compare_queries(graph, test_queries)
