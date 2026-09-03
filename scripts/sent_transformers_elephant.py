from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from utils import build_output_path, build_similarity_records, load_animal_word_pairs, save_results_to_csv


MODEL_NAMES = [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
CSV_DATA_PATH = "animals_oewn.csv"
PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"


def embed_sentence(model: SentenceTransformer, sentence: str) -> List[float]:
    """Generate embedding for one sentence using sentence-transformers."""
    embedding = model.encode(sentence, convert_to_numpy=True, show_progress_bar=False)
    return embedding.tolist()


def generate_word_pair_records(
    word_pairs: List[Dict[str, str]],
    model_name: str,
    sentence_template: str = "Word: {word} | Definition: {definition}",
) -> List[Dict[str, Any]]:
    """Generate similarity records for all word pairs using a sentence-transformers model."""
    model = SentenceTransformer(model_name)
    embed_fn = lambda sentence: embed_sentence(model, sentence)
    return build_similarity_records(word_pairs, embed_fn=embed_fn, sentence_template=sentence_template)


def run_models(
    models: List[str] | None = None,
    csv_path: str = CSV_DATA_PATH,
    sentence_template: str = "Word: {word} | Definition: {definition}",
) -> List[str]:
    """Run sentence-transformer similarity generation for each model and save a separate CSV per model."""
    selected_models = models or MODEL_NAMES
    animal_pairs = load_animal_word_pairs(str(PROJECT_DIR / csv_path))
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    output_paths: List[str] = []

    for model_name in selected_models:
        output_path = str(EMBEDDINGS_DIR / build_output_path(model_name=model_name))
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
                results, model_name=model_name, output_path=output_path
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
