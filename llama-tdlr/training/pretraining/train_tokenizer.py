from datasets import load_dataset, load_from_disk, concatenate_datasets
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from fire import Fire
from transformers import AutoTokenizer
import os
from glob import glob
import chardet
from tqdm import tqdm


def train_tokenizer(model_name_or_path, dataset_name=None, dataset_config_name=None,
                    output_dir=None, data_files=False, vocab_size=50265):
    # load dataset
    if data_files:
        data_files = {"train": glob(
            "/lustre07/scratch/gagan30/arocr/LLama/data/LLama_Data/Ar_Tok_data/*.txt")[:5]}
        print(data_files)
        print("Loading dataset from files")
        print(len(data_files["train"]))
        for filename in tqdm(data_files["train"]):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    _ = file.read()
            except UnicodeDecodeError:
                print(filename)

        dataset0 = load_dataset("text", split="train",
                                cache_dir="../cache/", data_files=data_files)
        dataset1 = load_from_disk(
            "/lustre07/scratch/gagan30/arocr/datasets/NewArOCRDatasetv5")['train']

        dataset = concatenate_datasets([dataset0, dataset1])
    else:
        dataset = load_from_disk(
            "/lustre07/scratch/gagan30/arocr/datasets/NewArOCRDatasetv5")['train']

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    # Customized training
    tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=vocab_size)

    # Save files to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    Fire(train_tokenizer)
