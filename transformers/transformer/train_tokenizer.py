import json

import sentencepiece as spm
from concurrent.futures import ProcessPoolExecutor
import os
from utils import make_dirs
from config import train_args, model_args


def get_mt_pairs(data_dir: str, splits=["train", "dev", "test"]):
    english_sentences = []
    chinese_sentences = []

    """
    json content:
    [["english sentence", "中文语句"], ["english sentence", "中文语句"]]
    """
    for split in splits:
        with open(f"{data_dir}/{split}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for pair in data:
                english_sentences.append(pair[0] + "\n")
                chinese_sentences.append(pair[1] + "\n")

    assert len(chinese_sentences) == len(english_sentences)

    print(f"the total number of sentences: {len(chinese_sentences)}")

    return chinese_sentences, english_sentences


def train_sentencepice_bpe(
    input_file: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float = 0.9995,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
):
    cmd = f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage={character_coverage} --pad_id={pad_id} --unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id}"
    spm.SentencePieceTrainer.train(cmd)


def train_tokenizer(
    source_corpus_path: str,
    target_corpus_path: str,
    source_vocab_size: int,
    target_vocab_size: int,
    source_character_coverage: float = 1.0,
    target_character_coverage: float = 0.9995,
) -> None:
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                train_sentencepice_bpe,
                source_corpus_path,
                "model_storage/source",
                source_vocab_size,
                source_character_coverage,
            ),
            executor.submit(
                train_sentencepice_bpe,
                target_corpus_path,
                "model_storage/target",
                target_vocab_size,
                target_character_coverage,
            ),
        ]

        for future in futures:
            future.result()

    sp = spm.SentencePieceProcessor()

    source_text = """
        Tesla is recalling nearly all 2 million of its cars on US roads to limit the use of its 
        Autopilot feature following a two-year probe by US safety regulators of roughly 1,000 crashes 
        in which the feature was engaged. The limitations on Autopilot serve as a blow to Tesla’s efforts 
        to market its vehicles to buyers willing to pay extra to have their cars do the driving for them.
        """

    sp.load("model_storage/source.model")
    print(sp.encode_as_pieces(source_text))
    ids = sp.encode_as_ids(source_text)
    print(ids)
    print(sp.decode_ids(ids))

    target_text = """
        新华社北京1月2日电（记者丁雅雯、李唐宁）2024年元旦假期，旅游消费十分火爆。旅游平台数据显示，旅游相关产品订单量大幅增长，“异地跨年”“南北互跨”成关键词。
        业内人士认为，元旦假期旅游“开门红”彰显消费潜力，预计2024年旅游消费有望保持上升势头。
    """

    sp.load("model_storage/target.model")
    print(sp.encode_as_pieces(target_text))
    ids = sp.encode_as_ids(target_text)
    print(ids)
    print(sp.decode_ids(ids))


if __name__ == "__main__":
    make_dirs(train_args.save_dir)

    chinese_sentences, english_sentences = get_mt_pairs(
        data_dir=train_args.dataset_path, splits=["train", "dev", "test"]
    )

    with open(f"{train_args.dataset_path}/corpus.ch", "w", encoding="utf-8") as f:
        f.writelines(chinese_sentences)

    with open(f"{train_args.dataset_path}/corpus.en", "w", encoding="utf-8") as f:
        f.writelines(english_sentences)

    train_tokenizer(
        f"{train_args.dataset_path}/corpus.en",
        f"{train_args.dataset_path}/corpus.ch",
        source_vocab_size=model_args.source_vocab_size,
        target_vocab_size=model_args.target_vocab_size,
    )
