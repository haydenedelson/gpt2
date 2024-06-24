from functools import partial
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

LOCAL_DIR = "edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8) # 100M tokens per shard, total of 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)


def tokenize(doc, tokenizer, eot_token):
    """Tokenize a document and rturn a numpy array of tokens"""
    tokens = [eot_token]
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == "__main__":
    # make data dir if does not exist
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu",
                      name=REMOTE_NAME,
                      split="train",
                      download_config=DownloadConfig(num_proc=8))
    print("loaded dataset")

    # init tokenizer
    encoder = tiktoken.get_encoding('gpt2')
    eot = encoder._special_tokens['<|endoftext|>']

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # allocate buffer to hold shard
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)

        token_count = 0
        pbar = None
        func = partial(tokenize, tokenizer=encoder, eot_token=eot)
        for tokens in pool.imap(func, fw, chunksize=16):
            # if updating current shard
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

                # update progress bar
                if pbar is None:
                    pbar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
                pbar.update(len(tokens))
            else:
                # write current shard and start new one
                split = "val" if shard_index == 0 else "train"
                file_name = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

                # write as much as possible from current document to this shard
                # any leftover will go into the next shard
                remainder = SHARD_SIZE - token_count
                pbar.update(remainder)
                all_tokens_np[token_count : token_count+remainder] = tokens[:remainder]
                write_datafile(file_name, all_tokens_np)
                shard_index += 1
                pbar = None

                # add leftovers to next shard
                all_tokens_np[0 : len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write remaining tokens as last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            file_name = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(file_name, all_tokens_np[:token_count])


