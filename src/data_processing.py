import fitz
import re
import unicodedata
import html
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional

def preprocess_text(pdf_path: str) -> Optional[str]:
    print(f"   - Lancement du Prétraitement pour le fichier : {pdf_path} ---")
    print(f"   - Extraction du texte brut avec PyMuPDF...")
    try:
        doc = fitz.open(pdf_path)
        raw_text = "".join([page.get_text("text", sort=True) for page in doc])
        doc_page_count = doc.page_count
        doc.close()
        print(f"   - Texte brut extrait par PyMuPDF (avant tout nettoyage) : {len(raw_text.split())} mots.")
        if not raw_text.strip():
            print("❌ AVERTISSEMENT : Le PDF ne contient aucun texte extractible.")
            return None
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE lors de la lecture du PDF : {e}")
        return None

    text = html.unescape(raw_text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = unicodedata.normalize("NFKC", text)

    end_sections_pattern = r'\n\s*(REFERENCES|ACKNOWLEDGEMENTS?|APPENDIX|BIBLIOGRAPHY)\s*\n'
    match = re.search(end_sections_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = text[:match.start()]
        print(f"   - Truncation avant la section '{match.group(1)}'.")
    else:
        print("   - Aucune section de fin majeure (Références, Annexes) trouvée.")

    copyright_pattern = r'\n\s*The Author\(s\) \d{4}.*?(?:\n\s*Open Access This chapter is licensed.*)?$'
    license_pattern = r'\n\s*Open Access This chapter is licensed under the terms of the Creative Commons Attribution-NonCommercial 4\.0 International License.*$'

    text = re.sub(license_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(copyright_pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()

    lines = text.split('\n')
    line_counts = Counter(
        line.strip() for line in lines
        if 5 < len(line.strip()) < 100 and not line.strip().isdigit() and not line.strip().isupper()
    )
    frequent_lines = {line for line, count in line_counts.items() if doc_page_count > 0 and count > (doc_page_count / 2) }

    if frequent_lines:
        print(f"   - Suppression de {len(frequent_lines)} lignes d'en-tête/pied de page fréquentes.")
        cleaned_text = '\n'.join(line for line in lines if line.strip() not in frequent_lines)
    else:
        cleaned_text = text

    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', cleaned_text)

    print("✅ Prétraitement terminé avec succès.")
    print(f"   - Texte final prétraité contient {len(cleaned_text.split())} mots.")

    return cleaned_text

def filter_irrelevant_chunks(chunks: list[str]) -> list[str]:
    keywords_to_exclude = [
        "creative commons", "license", "springer", "rights reserved", "doi", "chapter",
        "permitted use", "noncommercial", "creativecommons",
        "fig.", "figure", "table", "tableau", "annex", "appendice"
    ]
    filtered_chunks = [chunk for chunk in chunks if not any(kw in chunk.lower() for kw in keywords_to_exclude)]
    return [chunk for chunk in filtered_chunks if len(chunk.split()) > 50]

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

    final_filtered_chunks = filter_irrelevant_chunks(chunks)

    if len(final_filtered_chunks) < len(chunks):
        print(f"🧹 {len(chunks) - len(final_filtered_chunks)} chunks supprimés (bruit ou trop courts).")
    if not final_filtered_chunks:
        print("❌ AVERTISSEMENT : Aucun chunk pertinent n'a été conservé après le filtrage.")

    print(f"\n--- Affichage des {min(3, len(final_filtered_chunks))} premiers chunks filtrés ---\n") # Ajout d'un saut de ligne
    for i, chunk in enumerate(final_filtered_chunks[:3]):
        print(f"--- 🧩 Chunk {i+1} ({len(chunk.split())} mots) ---\n{chunk[:400]}...\n")

    return final_filtered_chunks
