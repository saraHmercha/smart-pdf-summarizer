from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from typing import List, Optional, Tuple, Dict, Any

# Dictionnaire pour stocker les modèles et tokenizers chargés en mémoire
_loaded_models_cache = {}

# --- Configuration des modèles Hugging Face (HF) ---
HF_MODELS_CONFIG = {
    "BART-large": {
        "model_name": "facebook/bart-large-cnn", # Entraîné spécifiquement pour le résumé
        "prefix": "" # BART n'a généralement pas besoin de préfixe
    },
    "T5-base": {
        "model_name": "t5-base",
        "prefix": "summarize: " # T5 bénéficie de ce préfixe
    },
    "Pegasus": {
        "model_name": "google/pegasus-cnn_dailymail", # Un bon Pegasus pour le résumé général
        "prefix": "" # Pegasus n'a généralement pas besoin de préfixe
    }
}

# --- Fonction de chargement de modèle HF ---
def load_hf_model_and_tokenizer(model_key: str, device: torch.device) -> Tuple[Any, Any]:
    """
    Charge un modèle Hugging Face et son tokenizer une seule fois et les met en cache.
    """
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

# --- Fonction de résumé pour un seul texte (HF ou OpenAI) ---
def summarize_text_abstractively(
    text: str,
    model_type: str, # "hf" ou "openai"
    hf_model_key: Optional[str] = None, # Clé du modèle HF (e.g., "BART-large")
    openai_api_key: Optional[str] = None,
    openai_model_name: str = "gpt-3.5-turbo",
    max_length: int = 150, # Longueur max du résumé généré
    min_length: int = 40,  # Longueur min du résumé généré
    device: Optional[torch.device] = None
) -> Optional[str]:
    """
    Génère un résumé abstractif pour un texte donné en utilisant soit un modèle Hugging Face,
    soit l'API OpenAI.
    """
    if not text or not text.strip():
        return None

    summary_result = None

    if model_type == "hf":
        if hf_model_key is None or hf_model_key not in HF_MODELS_CONFIG:
            print(f"❌ ERREUR : Clé de modèle HF non spécifiée ou non valide pour le type 'hf'.")
            return None
        
        model, tokenizer = load_hf_model_and_tokenizer(hf_model_key, device)
        if model is None or tokenizer is None:
            return None # Erreur de chargement déjà loggée

        input_text = HF_MODELS_CONFIG[hf_model_key]["prefix"] + text
        
        # --- CORRECTION DE LA LIGNE SUIVANTE ---
        # Fixe une limite d'entrée explicite pour éviter l'OverflowError
        # La limite typique pour BART est 1024, pour T5-base est 512
        # On peut la rendre dynamique si besoin en fonction du model_name
        fixed_max_input_length = 1024 if "bart" in hf_model_key.lower() else (512 if "t5" in hf_model_key.lower() else 1024)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=fixed_max_input_length, truncation=True)
        # --- FIN DE LA CORRECTION ---

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


# --- Fonction orchestrant le résumé abstrait d'un document entier (Map-Reduce) ---
def summarize_document_abstractively(
    extracted_chunks: List[str],
    hf_chunk_model_key: str, # Modèle HF pour résumer chaque chunk
    final_summary_model_type: str, # "hf" ou "openai"
    final_hf_model_key: Optional[str] = None, # Modèle HF pour le résumé final (si final_summary_model_type est "hf")
    openai_api_key: Optional[str] = None,
    openai_model_name: str = "gpt-3.5-turbo",
    max_chunk_summary_length: int = 100, # Longueur max des résumés de chunks
    max_final_summary_length: int = 400, # Longueur max du résumé final du document
    device: torch.device = None
) -> Optional[str]:
    """
    Orchestre le résumé abstractif d'un document long en utilisant une stratégie Map-Reduce.
    1. Résume chaque chunk extrait (Map).
    2. Concatène ces résumés de chunks.
    3. Résume le texte combiné pour obtenir le résumé final du document (Reduce).
    """
    if not extracted_chunks:
        print("❌ AVERTISSEMENT : La liste de chunks extraits est vide. Impossible de résumer abstractivement.")
        return None

    print(f"\n--- 🤖 Lancement du Résumé Abstractif (Map-Reduce) ---")

    # Étape 1 : Résumé abstractif de chaque chunk (Map)
    abstractive_chunk_summaries = []
    print(f"  - Résumé de chaque chunk avec {hf_chunk_model_key}...")
    for i, chunk_text in enumerate(extracted_chunks):
        print(f"    - Traitement du chunk {i+1}/{len(extracted_chunks)}...")
        chunk_abs_summary = summarize_text_abstractively(
            text=chunk_text,
            model_type="hf",
            hf_model_key=hf_chunk_model_key,
            max_length=max_chunk_summary_length,
            min_length=20,
            device=device
        )
        if chunk_abs_summary:
            abstractive_chunk_summaries.append(chunk_abs_summary)
        else:
            print(f"      - ⚠️ AVERTISSEMENT : Résumé abstractif échoué pour le chunk {i+1}. Ignoré.")

    if not abstractive_chunk_summaries:
        print("❌ Aucun résumé de chunk abstractif n'a pu être généré.")
        return None

    # Étape 2 : Concaténation des résumés de chunks
    combined_chunk_summaries = "\n\n".join(abstractive_chunk_summaries)
    print(f"  - Résumés de chunks combinés. Longueur totale : {len(combined_chunk_summaries.split())} mots.")

    # Étape 3 : Résumé final du document (Reduce)
    print(f"  - Génération du résumé final du document avec le modèle '{final_summary_model_type}'...")
    final_document_summary = summarize_text_abstractively(
        text=combined_chunk_summaries,
        model_type=final_summary_model_type,
        hf_model_key=final_hf_model_key,
        openai_api_key=openai_api_key,
        openai_model_name=openai_model_name,
        max_length=max_final_summary_length,
        min_length=50,
        device=device
    )

    if final_document_summary:
        print("✅ Résumé Abstractif Global terminé.")
        return final_document_summary
    else:
        print("❌ Échec de la génération du résumé abstractif global.")
        return None
