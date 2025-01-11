

from openai import OpenAI
import jsonlines
import random
import os
from tqdm import tqdm

from dotenv import load_dotenv

from utils import build_dataframe_from_csv

load_dotenv()

client = OpenAI()

def generate_similary_sentence(input_sentence: str) -> str:
    messages = [
        {"role": "system", "content": "你是中文专家"},
        {"role": "user", "content": f"生成一个与 '{input_sentence}' 意思一样的句子， 但基于不同的描述方式， 输出纯文本即可，不要添加引号。"}
    ]
    # 使用硅基流动
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="Qwen/Qwen2.5-7B-Instruct",  
        max_tokens=100,
        temperature=0.7,
    )

    return chat_completion.choices[0].message.content.strip()


def generate_train_data(data_path: str, negative_num: int = 15) -> None:
    # 加载原始数据
    df = build_dataframe_from_csv(data_path)

    base_name, _ = os.path.splitext(data_path)

    output_path = f"{base_name}.json"


    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating training data"):
        query1 = row["query1"]
        query2 = row["query2"]
        label = int(row["label"]) 

        neg_candidates = df["query2"].tolist()
        # 正样本
        if label == 1:
            pos = query2
        else:
            pos = generate_similary_sentence(query1)
        
        # 随机抽取负样本
        neg_candidates.remove(query2)  # 确保 query2 不在负样本中
        neg_samples = random.sample(neg_candidates, negative_num)
        if label == 0:
            # 如果 label 为 0，将 query2 放入负样本列表中
            neg_samples.insert(0, query2)  
            # 同时移除一个样本，保证负样本的数量一致
            neg_samples.pop()

        assert len(neg_samples) == negative_num
            

        # 构造数据
        result = {
            "query": query1,
            "pos": pos,
            "neg": neg_samples,
        }
        results.append(result)

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(results)

    print(f"Training data saved to {output_path}")

