from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import jsonlines
import random
import os

from utils import build_dataframe_from_csv


def load_model(model_name_or_path, device: str = None):
    return SentenceTransformer(model_name_or_path, device=device)


def get_corpus(data_path: str):
    queries = []
    candidates = []
    labels = []

    df = build_dataframe_from_csv(data_path)
    for _, row in df.iterrows():
        query1 = row["query1"]
        query2 = row["query2"]
        label = int(row["label"])

        queries.append(query1)
        candidates.append(query2)
        labels.append(label)

    return (queries, candidates, labels)


def get_jsonlines(json_path: str) -> list[dict]:
    jsons = []
    with jsonlines.open(json_path, mode="r") as reader:
        for obj in reader:
            jsons.append(obj)

    return jsons


def batch_search(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = 300,
    batch_size: int = 128,
):
    all_scores = []
    all_indices = []

    print(query_embeddings.shape)

    num_examples = query_embeddings.shape[0]

    for idx in tqdm(range(0, num_examples, batch_size), desc="Batches"):
        batch_queries = query_embeddings[idx : idx + batch_size]
        scores, indices = index.search(batch_queries, top_k)
        all_scores.extend(scores.tolist())
        all_indices.extend(indices.tolist())

    return all_scores, all_indices


def indexing(embeddings: np.ndarray, index_path: str) -> faiss.Index:
    """
    创建或加载Faiss索引。
    :param embeddings: 嵌入向量数组 (numpy.ndarray)
    :param index_path: 索引保存的路径
    :return: faiss.Index
    """
    if os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        print(f"Creating a new index and saving to {index_path}")
        dimension = embeddings.shape[1]  # 嵌入向量的维度
        index = faiss.IndexFlatL2(dimension)  # L2 距离索引
        index.add(embeddings)  # 添加嵌入向量到索引
        faiss.write_index(index, index_path)
    return index


def save_embeddings(embeddings: np.ndarray, file_path: str):
    """
    保存嵌入到文件
    :param embeddings: 嵌入向量 (numpy.ndarray)
    :param file_path: 保存路径
    """
    np.save(file_path, embeddings)
    print(f"Embeddings saved to {file_path}")


def load_embeddings(file_path: str) -> np.ndarray:
    """
    从文件加载嵌入
    :param file_path: 文件路径
    :return: 嵌入向量 (numpy.ndarray)
    """
    embeddings = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embeddings


def generate_hard_negatives(
    model_name,
    data_path: str,
    output_path: str,
    json_path: str,
    embedding_batch_size: int = 128,
    negative_num: int = 15,
    device: str = None,
):
    model = load_model(model_name, device)
    queries, candidates, labels = get_corpus(data_path)

    candidate_embeddings = None

    candidate_embeddings_path = "embeddings.faiss"
    if not os.path.exists(candidate_embeddings_path):
        candidate_embeddings = model.encode(
            candidates,
            batch_size=embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    index = indexing(candidate_embeddings, candidate_embeddings_path)

    query_embeddings_path = "queries.npy"
    if os.path.exists(query_embeddings_path):
        query_embeddings = load_embeddings(query_embeddings_path)
    else:
        query_embeddings = model.encode(
            queries,
            batch_size=embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        save_embeddings(query_embeddings, query_embeddings_path)

    _, all_indices = batch_search(index, query_embeddings, top_k=300)

    jsons = get_jsonlines(json_path)

    results = []

    for query_idx, query in enumerate(queries):
        indices = all_indices[query_idx]  # topk 的索引列表
        # 可采样的候选索引列表
        candidate_indices = []
        for idx in indices[5:]:
            if candidates[idx] not in [jsons[query_idx]["pos"], query]:
                candidate_indices.append(idx)

        neg_samples = [
            candidates[i] for i in random.sample(candidate_indices, negative_num)
        ]

        result = {
            "query": query,
            "pos": jsons[query_idx]["pos"],
            "neg": neg_samples,
        }
        results.append(result)

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(results)

    print(f"Training data saved to {output_path}")


if __name__ == "__main__":
    generate_hard_negatives(
        "/home/user/greyfoss/workspace/nlp-in-action-public/text_matching/sentence-berts/bi-encoder/cosent-loss/output/hfl-chinese-macbert-large-2025-01-13_15-18-07",
        data_path="data/train.txt",
        output_path="data/train.json",
        json_path="data/train_simple.json",
        negative_num=15,
        device="cuda:2",
    )
