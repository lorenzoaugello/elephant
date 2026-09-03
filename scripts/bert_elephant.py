from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from utils import build_output_path, build_similarity_records, load_animal_word_pairs, save_results_to_csv


MODEL_NAMES = [
    "bert-base-uncased",
    "bert-base-cased",
    "FacebookAI/roberta-base",
    "FacebookAI/xlm-roberta-base"
    
]
CSV_DATA_PATH = "animals_oewn.csv"
CSV_OUTPUT_PREFIX = "bert_similarity_results"
PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"


def load_bert_model(model_name: str):
    """Load tokenizer and model for a Hugging Face BERT model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_sentence(tokenizer, model, sentence: str) -> List[float]:
    """Compute a mean-pooled BERT embedding for one sentence."""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
    summed_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    embedding = (pooled / summed_mask).squeeze(0).cpu().tolist()
    return embedding


def generate_word_pair_records(
    word_pairs: List[Dict[str, str]],
    model_name: str,
    sentence_template: str = "Word: {word} | Definition: {definition}",
) -> List[Dict[str, Any]]:
    """Generate similarity records for all word pairs with a Hugging Face BERT model."""
    tokenizer, model = load_bert_model(model_name)
    embed_fn = lambda sentence: embed_sentence(tokenizer, model, sentence)
    return build_similarity_records(word_pairs, embed_fn=embed_fn, sentence_template=sentence_template)


def run_models(
    models: List[str] | None = None,
    csv_path: str = CSV_DATA_PATH,
    sentence_template: str = "Word: {word} | Definition: {definition}",
) -> List[str]:
    """Run BERT similarity generation for each model and save a separate CSV per model."""
    selected_models = models or MODEL_NAMES
    animal_pairs = load_animal_word_pairs(str(PROJECT_DIR / csv_path))
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    output_paths: List[str] = []

    for model_name in selected_models:
        output_path = str(EMBEDDINGS_DIR / build_output_path(model_name=model_name, prefix=CSV_OUTPUT_PREFIX))
        output_file = Path(output_path)

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[{model_name}] Found existing output file: {output_file}. Skipping generation.")
            output_paths.append(output_path)
            continue

        try:
            results = generate_word_pair_records(
                animal_pairs,
                model_name=model_name,
                sentence_template=sentence_template,
            )
            output_path = save_results_to_csv(
                results, model_name=model_name, output_path=output_path, prefix=CSV_OUTPUT_PREFIX
            )
            output_paths.append(output_path)
            print(f"[{model_name}] Loaded {len(animal_pairs)} animal pairs from {csv_path}")
            print(f"[{model_name}] Saved {len(results)} similarity records to {output_path}")
        except Exception as exc:
            print(f"[{model_name}] Error generating similarity results: {exc}")
            raise

    return output_paths


if __name__ == "__main__":
    run_models()
