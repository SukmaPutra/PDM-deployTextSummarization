# app_streamlit.py

import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import re
import os
import torch # Tambahkan untuk model abstractive
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Tambahkan untuk model abstractive
from bs4 import BeautifulSoup # Tambahkan untuk preprocessing abstractive

print("--- Memulai Aplikasi Streamlit ---")

# --- Pastikan NLTK resources terunduh ---
print("Memeriksa dan mengunduh sumber daya NLTK (punkt, stopwords)...")
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
print("Sumber daya NLTK siap.")

# --- Path ke Model ---
W2V_MODEL_PATH = "word2vec_model.bin"
ABSTRACTIVE_MODEL_PATH = "./abstractive_model_artifacts" # Path baru untuk model abstractive

# --- Fungsi Preprocessing untuk TextRank (Extractive) ---
def preprocess_single_article_extractive(article_content):
    cleaned_sentences_tokenized = []
    original_relevant_sentences = []
    stop_words = set(stopwords.words('indonesian'))

    sentences_from_content = sent_tokenize(article_content)

    for sentence in sentences_from_content:
        cleaned_sent_for_analysis = re.sub(r'\[baca:\s*[^\]]*\]', '', sentence, flags=re.IGNORECASE)
        cleaned_sent_for_analysis = re.sub(r'advertisement', '', cleaned_sent_for_analysis, flags=re.IGNORECASE)

        words = [word.lower() for word in word_tokenize(cleaned_sent_for_analysis)]
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

        if filtered_words:
            cleaned_sentences_tokenized.append(filtered_words)
            original_relevant_sentences.append(sentence)
    return cleaned_sentences_tokenized, original_relevant_sentences

# --- Fungsi untuk mendapatkan vektor kalimat (Extractive) ---
def get_sentence_vector(sentence_tokens, word2vec_model):
    if word2vec_model is None:
        return np.zeros(200) # Ukuran vektor harus sesuai dengan yang dilatih

    vectors = []
    for word in sentence_tokens:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(word2vec_model.vector_size)

# --- Fungsi TextRank untuk Peringkasan (Extractive) ---
def textrank_summarize(article_content, word2vec_model, target_word_count):
    all_original_sentences_in_order = sent_tokenize(article_content)
    fallback_summary = " ".join(all_original_sentences_in_order[:min(len(all_original_sentences_in_order), 2)])

    cleaned_sentences_tokenized, original_relevant_sentences = preprocess_single_article_extractive(article_content)

    if not cleaned_sentences_tokenized or len(cleaned_sentences_tokenized) < 2:
        return fallback_summary, "Gagal (Teks Sangat Pendek)"

    sentence_vectors = []
    valid_original_indices = []

    for i, tokens in enumerate(cleaned_sentences_tokenized):
        vec = get_sentence_vector(tokens, word2vec_model)
        if vec is not None and not np.all(vec == 0):
            sentence_vectors.append(vec)
            valid_original_indices.append(i)

    if len(sentence_vectors) < 2:
        return fallback_summary, "Gagal (Vektor Kurang)"

    try:
        similarity_matrix = cosine_similarity(sentence_vectors)
    except ValueError:
        return fallback_summary, "Gagal (Masalah Dimensi Vektor)"

    graph = nx.from_numpy_array(similarity_matrix)
    scores = {}
    try:
        scores = nx.pagerank(graph, max_iter=1000, tol=1e-3)
    except nx.PowerIterationFailedConvergence:
        st.warning("Peringatan: PageRank gagal konvergen untuk teks ini. Menggunakan fallback.")
        return fallback_summary, "Gagal (PageRank Konvergensi)"

    ranked_processed_indices = sorted(((scores[i], idx_in_valid) for idx_in_valid, i in enumerate(scores)), reverse=True)

    current_word_count = 0
    final_summary_pairs = []
    num_selected_sentences = 0

    for score, processed_idx_in_valid in ranked_processed_indices:
        original_idx_in_relevant = valid_original_indices[processed_idx_in_valid]
        original_sentence = original_relevant_sentences[original_idx_in_relevant]

        try:
            actual_original_idx = all_original_sentences_in_order.index(original_sentence)

            if original_sentence not in [s for _, s in final_summary_pairs]:
                final_summary_pairs.append((actual_original_idx, original_sentence))
                current_word_count += len(word_tokenize(original_sentence))
                num_selected_sentences += 1
        except ValueError:
            pass

        if current_word_count >= target_word_count or num_selected_sentences >= len(original_relevant_sentences) * 0.5 + 1:
            break

    if not final_summary_pairs and len(all_original_sentences_in_order) > 0:
        return fallback_summary, "Gagal (Ringkasan Kosong/Terlalu Pendek)"

    final_summary_sentences_ordered = [sent for idx, sent in sorted(final_summary_pairs)]

    return " ".join(final_summary_sentences_ordered), "Sukses"

# --- Fungsi Pra-pemrosesan untuk Abstractive ---
def preprocess_abstractive(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Fungsi Peringkasan Abstractive dengan IndoBART ---
def indobart_summarize(text, tokenizer, model, min_len=30, max_len=150):
    clean_text = preprocess_abstractive(text)

    # Periksa dan batasi panjang input jika terlalu panjang
    # Tokenizer akan memotong jika max_length terlampaui
    if len(clean_text.split()) > 512: # Estimasi kasar
        st.warning("Peringatan: Teks input sangat panjang. Model abstractive mungkin hanya memproses bagian awalnya.")

    with torch.no_grad(): # Nonaktifkan perhitungan gradien untuk inferensi
        input_ids = tokenizer.encode(clean_text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(
            input_ids,
            min_length=min_len,
            max_length=max_len,
            num_beams=2,
            repetition_penalty=2.0,
            length_penalty=0.8,
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, "Sukses"


# --- Pemuatan Model (dengan st.cache_resource) ---
# Word2Vec untuk TextRank
@st.cache_resource
def load_word2vec_model(path=W2V_MODEL_PATH):
    st.spinner("Memuat model Word2Vec (Extractive)...")
    if os.path.exists(path):
        try:
            model = Word2Vec.load(path)
            st.success("Model Word2Vec berhasil dimuat!")
            return model
        except Exception as e:
            st.error(f"Gagal memuat model Word2Vec: {e}. Pastikan file '{path}' ada dan tidak rusak.")
            st.info("Anda mungkin perlu melatih dan menyimpan model Word2Vec terlebih dahulu.")
            return None
    else:
        st.error(f"File model Word2Vec '{path}' tidak ditemukan.")
        st.info("Harap pastikan Anda telah menempatkan file ini di folder yang benar.")
        return None

# IndoBART untuk Abstractive
@st.cache_resource
def load_indobart_model(path=ABSTRACTIVE_MODEL_PATH):
    st.spinner("Memuat model IndoBART (Abstractive)...")
    if os.path.exists(path) and os.path.isdir(path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            model.eval()
            st.success("Model IndoBART berhasil dimuat!")
            return tokenizer, model
        except Exception as e:
            st.error(f"Gagal memuat model IndoBART: {e}. Pastikan artefak model ada di '{path}'.")
            st.info("Anda perlu mengunduh dan menyimpan model IndoBART ke folder ini terlebih dahulu.")
            return None, None
    else:
        st.error(f"Folder model IndoBART '{path}' tidak ditemukan atau tidak lengkap.")
        st.info("Harap pastikan Anda telah mengunduh dan menyimpan model IndoBART ke folder ini.")
        return None, None

# Panggil fungsi untuk memuat kedua model saat aplikasi dimulai
word2vec_model = load_word2vec_model()
indobart_tokenizer, indobart_model = load_indobart_model()


# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(
    page_title="Text Summarizer (Extractive & Abstractive)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📝 Text Summarizer (Extractive & Abstractive)")
st.markdown("Aplikasi ini memungkinkan Anda memilih antara peringkasan *extractive* (TextRank) dan *abstractive* (IndoBART).")
st.markdown("---")

# Input teks dari pengguna
user_input_text = st.text_area(
    "Masukkan teks yang ingin diringkas di sini:",
    height=300,
    placeholder="Contoh: Presiden Susilo Bambang Yudhoyono siang nanti dijadwalkan berpidato di hadapan para prajurit dan perwira TNI di Markas Besar TNI Cilangkap, Jakarta Timur. Pidato Presiden terkait dengan peringatan Hari Ulang Tahun ke-60 TNI. Keterangan ini disampaikan Panglima TNI Marsekal Djoko Suyanto di Jakarta, Kamis (5/10). Yudhoyono rencananya akan tiba di Cilangkap sekitar pukul 14.00 WIB. Selain presiden, purnawirawan TNI serta keluarga prajurit juga akan menghadiri acara yang digelar di Lapangan Udara Cilangkap ini. Menurut Djoko, pidato ini merupakan kegiatan rutin yang dilakukan presiden setiap peringatan hari ulang tahun TNI. Selain itu juga sebagai sarana komunikasi antarpimpinan negara dan militer. (DNP/Tim Liputan 6 SCTV)"
)

# Pilihan model peringkasan
summary_type = st.radio(
    "Pilih Jenis Peringkasan:",
    ("Extractive (TextRank)", "Abstractive (IndoBART)"),
    help="*Extractive*: Memilih kalimat kunci dari teks asli. *Abstractive*: Membuat kalimat baru yang meringkas ide utama."
)

summary_length_options = {}

# Slider untuk mengatur panjang ringkasan berdasarkan jenis model
if summary_type == "Extractive (TextRank)":
    target_word_count_slider = st.slider(
        "Panjang Ringkasan Target (Jumlah Kata untuk TextRank)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Pilih jumlah kata yang Anda inginkan untuk ringkasan. Model akan mencoba mendekati jumlah ini."
    )
    summary_length_options['extractive_word_count'] = target_word_count_slider
elif summary_type == "Abstractive (IndoBART)":
    col1, col2 = st.columns(2)
    with col1:
        abs_min_length = st.slider(
            "Panjang Minimum (Kata untuk IndoBART)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Jumlah kata minimum untuk ringkasan abstrak."
        )
    with col2:
        abs_max_length = st.slider(
            "Panjang Maksimum (Kata untuk IndoBART)",
            min_value=50,
            max_value=300,
            value=150,
            step=5,
            help="Jumlah kata maksimum untuk ringkasan abstrak."
        )
    summary_length_options['abstractive_min_length'] = abs_min_length
    summary_length_options['abstractive_max_length'] = abs_max_length


# Tombol untuk memulai proses ringkasan
if st.button("Ringkas Teks"):
    if not user_input_text.strip():
        st.warning("Mohon masukkan teks terlebih dahulu untuk diringkas.")
    else:
        summary = ""
        status = "Gagal"

        if summary_type == "Extractive (TextRank)":
            if word2vec_model is None:
                st.error("Model Word2Vec tidak dapat dimuat. Proses peringkasan TextRank tidak dapat dilakukan.")
            else:
                with st.spinner("Meringkas teks Anda dengan TextRank..."):
                    summary, status = textrank_summarize(
                        user_input_text,
                        word2vec_model,
                        summary_length_options['extractive_word_count']
                    )
        elif summary_type == "Abstractive (IndoBART)":
            if indobart_tokenizer is None or indobart_model is None:
                st.error("Model IndoBART tidak dapat dimuat. Proses peringkasan Abstractive tidak dapat dilakukan.")
            else:
                with st.spinner("Meringkas teks Anda dengan IndoBART..."):
                    summary, status = indobart_summarize(
                        user_input_text,
                        indobart_tokenizer,
                        indobart_model,
                        summary_length_options['abstractive_min_length'],
                        summary_length_options['abstractive_max_length']
                    )
        
        st.subheader("Hasil Ringkasan:")
        if status == "Sukses":
            st.success(summary)
            st.info(f"Jenis Ringkasan: **{summary_type}** | Total Kata Ringkasan: **{len(word_tokenize(summary))}** | Status: **{status}**")
        else:
            st.error(f"Gagal membuat ringkasan. Status: {status}")
            st.write(summary) # Menampilkan fallback jika ada

st.markdown("---")
st.markdown("Dibuat dengan ❤️ ")