import fitz
import re
import unicodedata
import html
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional

def preprocess_text(pdf_path: str) -> Optional[str]:
    print(f"   - Lancement du Prétraitement pour le fichier : {pdf_path} ---")
    print(f"   - Extraction du texte brut...")
    try:
        doc = fitz.open(pdf_path)
        raw_text = "".join([page.get_text("text", sort=True) for page in doc])
        doc.close()
        if not raw_text.strip():
            print("❌ AVERTISSEMENT : Le PDF ne contient aucun texte extractible.")
            return None
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors de la lecture du PDF : {e}")
        return None

    # --- NOUVELLE LOGIQUE POUR GÉRER LA SUPPRESSION DES SECTIONS DE FIN ---
    # Convertit les entités HTML, normalise les guillemets et gère les coupures de mots
    text = html.unescape(raw_text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'-\n', '', text) # Gère les mots coupés en fin de ligne

    # Appliquez les split sur le texte brut pour isoler la partie principale du contenu
    # Gardez la partie principale du texte et retirez les sections de fin
    
    # Motif pour les sections de fin courantes (References, Acknowledgements, Appendix)
    # Utilisez re.split et prenez la première partie, qui est le contenu principal
    main_content_end_pattern = r'\n\s*(References|Acknowledgements?|Appendix|Bibliography)\s*\n'
    match = re.search(main_content_end_pattern, text, flags=re.IGNORECASE)
    if match:
        text = text[:match.start()] # Prend tout le texte AVANT le début de la section trouvée

    # Motifs spécifiques de copyright/licence qui peuvent être à la fin du document
    copyright_pattern = r'The Author\(s\) \d{4}.*'
    license_pattern = r'Open Access This chapter is licensed under the terms of the Creative Commons Attribution-NonCommercial 4\.0 International License.*'
    
    text = re.split(copyright_pattern, text, flags=re.DOTALL)[0] if re.search(copyright_pattern, text, flags=re.DOTALL) else text
    text = re.split(license_pattern, text, flags=re.DOTALL)[0] if re.search(license_pattern, text, flags=re.DOTALL) else text
    # --- FIN NOUVELLE LOGIQUE ---

    # Nettoyage général du texte (appliqué APRÈS la suppression des sections de fin)
    text = re.sub(r'\n\s*\n', '\n\n', text)        # Paragraphes vides multiples en un seul
    text = re.sub(r'[ \t]+', ' ', text)             # Espaces et tabulations multiples
    text = unicodedata.normalize("NFKC", text)      # Normalisation Unicode

    # Suppression des entêtes/pieds répétitifs (peut être plus robuste si fait par page)
    # Cette heuristique est à garder car elle attrape les entêtes qui ne sont pas des sections
    lines = text.split('\n')
    line_counts = Counter(
        line.strip() for line in lines
        if 10 < len(line.strip()) < 100 and not line.strip().isdigit() and not line.strip().isupper()
    )
    frequent_lines = {line for line, count in line_counts.items() if count > 2}

    if frequent_lines:
        print(f"   - Suppression de {len(frequent_lines)} lignes d'en-tête/pied de page fréquentes.")
        cleaned_text = '\n'.join(line for line in lines if line.strip() not in frequent_lines)
    else:
        cleaned_text = text

    # Nettoyage final des espaces et caractères invisibles
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', cleaned_text)
    
    print("✅ Prétraitement terminé avec succès.")
    return cleaned_text

def filter_irrelevant_chunks(chunks: list[str]) -> list[str]:
    keywords_to_exclude = [
        "creative commons", "license", "springer", "rights reserved", "doi", "chapter",
        "permitted use", "noncommercial", "creativecommons",
        "fig.", "figure", "table", "tableau", "annex", "appendice"
        # "references", "acknowledgment", "acknowledgement", "table of contents"
        # Ces termes sont maintenant gérés par la suppression des sections de fin,
        # ou ne devraient pas être exclus s'ils apparaissent dans le texte principal comme titres de section.
        # Laissez-les ici si vous voulez filtrer les mentions DANS les chunks.
    ]
    filtered_chunks = [chunk for chunk in chunks if not any(kw in chunk.lower() for kw in keywords_to_exclude)]
    return filtered_chunks

def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 128) -> list[str]:
    if not text:
        print("❌ AVERTISSEMENT : Texte vide fourni pour le découpage.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(f"✅ Découpage terminé. Nombre de chunks bruts créés : {len(chunks)}")

    filtered_chunks_by_keywords = filter_irrelevant_chunks(chunks)
    final_filtered_chunks = [chunk for chunk in filtered_chunks_by_keywords if len(chunk.split()) > 50]

    if len(final_filtered_chunks) < len(chunks):
        print(f"🧹 {len(chunks) - len(final_filtered_chunks)} chunks supprimés (bruit ou trop courts).")
    if not final_filtered_chunks:
        print("❌ AVERTISSEMENT : Aucun chunk pertinent n'a été conservé après le filtrage.")

    print(f"\n--- Affichage des {min(3, len(final_filtered_chunks))} premiers chunks filtrés ---")
    for i, chunk in enumerate(final_filtered_chunks[:3]):
        print(f"\n--- 🧩 Chunk {i+1} ({len(chunk.split())} mots) ---\n{chunk[:400]}...\n")

    return final_filtered_chunks
