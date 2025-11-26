from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any
import gc

model = AutoModel.from_pretrained('./contriever-msmarco', torch_dtype=torch.float16).eval().to('cuda')
tokenizer = AutoTokenizer.from_pretrained('./contriever-msmarco')


def split_text_into_blocks(text: str, max_block_length: int = 64) -> List[str]:
    words = text.split()
    blocks = []
    stride = max_block_length // 2

    for i in range(0, len(words), stride):
        block = " ".join(words[i:i + max_block_length])
        if block:
            blocks.append(block)

    return blocks


def get_context_from_hotpotqa(data_item) -> List[str]:
    context_blocks = []

    context = data_item['context']
    titles = context['title']
    paragraphs = context['sentences']

    for title, para_list in zip(titles, paragraphs):
        for para in para_list:
            if para:
                para_with_title = f"{title}: {para}"
                blocks = split_text_into_blocks(para_with_title)
                context_blocks.extend(blocks)

    return context_blocks


def retrieve_relevant_passages_hotpotqa(query_text: str, current_data, k: int = 5) -> List[str]:
    text_blocks = get_context_from_hotpotqa(current_data)

    if not text_blocks:
        return []

    block_embeddings = contriever_encode(text_blocks, batch_size=8)
    query_embedding = contriever_encode([query_text], batch_size=1)

    similarities = np.dot(block_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-k:][::-1]
    top_results = [text_blocks[i] for i in top_indices if i < len(text_blocks)]

    gc.collect()
    torch.cuda.empty_cache()

    return top_results


def contriever_encode(texts: List[str], batch_size: int = 8) -> np.ndarray:
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to('cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs["attention_mask"]
        batch_embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)

        all_embeddings.append(batch_embeddings.cpu().float().numpy())

        del inputs, outputs, batch_embeddings
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
    dataset = load_from_disk('./hotpotqa_validation')
    current_data = dataset[0]

    print(current_data['question'])
    print(current_data['answer'])
    print(retrieve_relevant_passages_hotpotqa(current_data['question'], current_data))