from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from typing import List, Optional, Tuple, Dict, Any

_loaded_models_cache = {}

HF_MODELS_CONFIG = {
    "BART-large": {
        "model_name": "facebook/bart-large-cnn",
        "prefix": ""
    },
    "T5-base": {
        "model_name": "t5-base",
        "prefix": "summarize: "
    },
    "Pegasus": {
        "model_name": "google/pegasus-cnn_dailymail",
        "prefix": ""
    }
}

def load_hf_model_and_tokenizer(model_key: str, device: torch.device) -> Tuple[Any, Any]:
    if model_key not in HF_MODELS_CONFIG:
        raise ValueError(f"Modèle HF '{model_key}' non configuré.")
    model_name = HF_MODELS_CONFIG[model_key]["model_name"]
    if model_name in _loaded_models_cache:
        return _loaded_models_cache[model_name]["model"], _loaded_models_cache[model_name]["tokenizer"]
    print(f"   - ⏳ Chargement du modèle '{model_name}' sur {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        _loaded_models_cache[model_name] = {"tokenizer": tokenizer, "model": model}
        print(f"   - ✅ Modèle '{model_name}' chargé.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ ERREUR lors du chargement de '{model_name}': {e}")
        return None, None

def summarize_text_abstractively(
    text: str,
    model_type: str,
    hf_model_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model_name: str = "gpt-3.5-turbo",
    max_length: int = 150,
    min_length: int = 40,
    device: Optional[torch.device] = None
) -> Optional[str]:
    if not text or not text.strip():
        return None
    summary_result = None
    if model_type == "hf":
        if hf_model_key is None or hf_model_key not in HF_MODELS_CONFIG:
            print(f"❌ ERREUR : Clé de modèle HF non spécifiée ou non valide pour le type 'hf'.")
            return None
        model, tokenizer = load_hf_model_and_tokenizer(hf_model_key, device)
        if model is None or tokenizer is None:
            return None
        input_text = HF_MODELS_CONFIG[hf_model_key]["prefix"] + text
        fixed_max_input_length = 1024
        if "bart" in hf_model_key.lower():
            fixed_max_input_length = 1024
        elif "t5" in hf_model_key.lower():
            fixed_max_input_length = 512
        elif "pegasus" in hf_model_key.lower():
            fixed_max_input_length = 1024
        inputs = tokenizer(input_text, return_tensors="pt", max_length=fixed_max_input_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        try:
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary_result = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"      - ❌ Erreur lors du résumé HF : {e}")
            summary_result = None
    elif model_type == "openai":
        if openai_api_key is None:
            print("❌ ERREUR : Clé API OpenAI non fournie pour le type 'openai'.")
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in summarizing technical documents and courses. Provide a concise, clear and comprehensive summary."},
                    {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
                ],
                max_tokens=max_length,
                temperature=0.7,
            )
            summary_result = response.choices[0].message.content.strip()
        except ImportError:
            print("❌ ERREUR : La librairie 'openai' n'est pas installée. `pip install openai`.")
            summary_result = None
        except Exception as e:
            print(f"❌ ERREUR lors de l'appel à l'API OpenAI : {e}")
            summary_result = None
    else:
        print(f"❌ ERREUR : Type de modèle '{model_type}' non supporté.")
        summary_result = None
    return summary_result if summary_result else None

def summarize_document_abstractively(
    extracted_chunks: List[str],
    hf_chunk_model_key: str,
    final_summary_model_type: str,
    final_hf_model_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model_name: str = "gpt-3.5-turbo",
    max_chunk_summary_length: int = 25,
    max_final_summary_length: int = 400,
    device: torch.device = None
) -> Optional[str]:
    if not extracted_chunks:
        print("❌ AVERTISSEMENT : La liste de chunks extraits est vide. Impossible de résumer abstractivement.")
        return None

    print(f"\n--- 🤖 Lancement du Résumé Abstractif (Map-Reduce) ---")

    abstractive_chunk_summaries = []
    print(f"  - Résumé de chaque chunk avec {hf_chunk_model_key}...")
    for i, chunk_text in enumerate(extracted_chunks):
        print(f"    - Traitement du chunk {i+1}/{len(extracted_chunks)}...")
        chunk_abs_summary = summarize_text_abstractively(
            text=chunk_text,
            model_type="hf",
            hf_model_key=hf_chunk_model_key,
            max_length=max_chunk_summary_length,
            min_length=10,
            device=device
        )
        if chunk_abs_summary:
            print(f"      - Résumé abstractif du chunk {i+1} : {len(chunk_abs_summary.split())} mots.")
            print(f"        Contenu (début) : \"{chunk_abs_summary[:100]}...\"")
            abstractive_chunk_summaries.append(chunk_abs_summary)
        else:
            print(f"      - ⚠️ AVERTISSEMENT : Résumé abstractif échoué pour le chunk {i+1}. Ignoré.")

    if not abstractive_chunk_summaries:
        print("❌ Aucun résumé de chunk abstractif n'a pu être généré.")
        return None

    combined_chunk_summaries = "\n\n".join(abstractive_chunk_summaries)
    print(f"  - Résumés de chunks combinés. Longueur totale : {len(combined_chunk_summaries.split())} mots.")

    print(f"  - Génération du résumé final du document avec le modèle '{final_summary_model_type}'...")
    final_document_summary = summarize_text_abstractively(
        text=combined_chunk_summaries,
        model_type=final_summary_model_type,
        hf_model_key=final_hf_model_key,
        openai_api_key=openai_api_key,
        openai_model_name=openai_model_name,
        max_length=max_final_summary_length,
        min_length=150,
        device=device
    )

    if final_document_summary:
        print("✅ Résumé Abstractif Global terminé.")
        return final_document_summary
    else:
        print("❌ Échec de la génération du résumé abstractif global.")
        return None
