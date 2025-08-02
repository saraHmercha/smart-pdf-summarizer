import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def extractive_summary(chunks: list[str], sentences_count: int = 3, language: str = "english") -> list[str]:
    print(f"--- ✂️ Lancement du Résumé Extactif (TextRank) pour {len(chunks)} chunks en {language} ---")

    try:
        stemmer = Stemmer(language)
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
    except LookupError:
        print(f"❌ ERREUR : La langue '{language}' n'est pas supportée par sumy ou les données NLTK associées sont manquantes.")
        print("Veuillez vous assurer que les données NLTK pour cette langue sont téléchargées.")
        print("Retour au résumé extractif sans stop words spécifiques ou en english par défaut.")
        stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

    extracted_chunks = []

    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            extracted_chunks.append("")
            continue
        try:
            parser = PlaintextParser.from_string(chunk, Tokenizer(language))
            num_actual_sentences = len(list(parser.document.sentences))
            sentences_to_extract = min(sentences_count, num_actual_sentences)

            if sentences_to_extract == 0:
                extracted_chunks.append("")
                continue

            summary_sentences = summarizer(parser.document, sentences_to_extract)
            summary_text = " ".join(str(sentence) for sentence in summary_sentences)
            extracted_chunks.append(summary_text)

        except Exception as e:
            print(f"⚠️ AVERTISSEMENT : Erreur lors du résumé extractif du chunk {i+1} (longueur {len(chunk.split())} mots) : {e}")
            extracted_chunks.append(chunk)

    print("✅ Résumé Extactif terminé.")
    return extracted_chunks
