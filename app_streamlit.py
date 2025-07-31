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
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import time

# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(
    page_title="Text Summarizer (Extractive & Abstractive)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
# Asumsi Word2Vec masih lokal. Jika ini juga diunggah, perlu diubah.
W2V_MODEL_PATH = "word2vec_model.bin" 

# Ganti ini dengan ID repositori Hugging Face Anda
# Ini adalah repositori yang baru saja Anda unggah ke Hugging Face Hub
ABSTRACTIVE_MODEL_ID = "SukmaPutra/abstractive_model_artifacts" 

# --- Fungsi untuk mengambil teks dari URL ---
def get_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Akan memunculkan HTTPError untuk status kode 4xx/5xx

        soup = BeautifulSoup(response.text, 'html.parser')

        # Hapus skrip dan style agar tidak mengganggu ekstraksi teks
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        article_text = ""
        
        # Opsi 1: Coba cari tag semantik atau class/id umum yang mengandung konten artikel
        possible_content_tags = [
            {'name': 'article'},
            {'name': 'main'},
            {'name': 'div', 'class_': re.compile(r'article|content|post|story|body|entry-content', re.IGNORECASE)},
            {'name': 'section', 'class_': re.compile(r'article|content|post|story|body|entry-content', re.IGNORECASE)},
        ]

        for selector in possible_content_tags:
            tag_name = selector['name']
            tag_attrs = {k:v for k,v in selector.items() if k!='name'}
            
            found = soup.find(tag_name, **tag_attrs)
            if found:
                article_text = found.get_text(separator=' ', strip=True)
                # Pastikan teks yang didapat cukup panjang dan relevan
                if len(article_text.split()) > 50: # Minimal 50 kata sebagai indikasi konten
                    break
                else: # Jika terlalu pendek, coba cari yang lain
                    article_text = ""
        
        # Opsi 2: Jika tidak ditemukan konten spesifik, ambil semua teks dari tag paragraf
        if not article_text or len(article_text.split()) < 50:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        # Opsi 3: Jika masih kosong atau sangat pendek, ambil semua teks dari body
        if not article_text or len(article_text.split()) < 50:
            if soup.body:
                article_text = soup.body.get_text(separator=' ', strip=True)
            else:
                st.warning("Tidak dapat menemukan tag <body> di halaman.")
                return None

        # Lakukan pembersihan tambahan pada teks yang diambil
        # Ganti spasi berlebih, tab, dan newline dengan satu spasi
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        # Hapus karakter non-ASCII yang tidak diinginkan (opsional, tergantung kebutuhan)
        # article_text = re.sub(r'[^\x00-\x7F]+', ' ', article_text) 
        
        # Filter kalimat yang terlalu pendek atau berisi boilerplate
        sentences = sent_tokenize(article_text)
        filtered_sentences = [
            s for s in sentences 
            if len(s.split()) > 5 and not re.search(r'^\s*(gambar|foto|video|iklan|advertisement|copyright|terkait|baca juga|ikuti kami|subscribe|newsletter|komentar|bagikan)\s*[:.]?\s*$', s, re.IGNORECASE)
        ]
        article_text = " ".join(filtered_sentences)

        return article_text if len(article_text.strip()) > 0 else None

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil artikel dari URL: {e}. Pastikan URL valid dan dapat diakses.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses URL: {e}")
        return None

# --- Fungsi validasi URL sederhana ---
def is_valid_url(url):
    try:
        result = urlparse(url)
        # Pastikan ada skema (http/https) dan lokasi jaringan (domain)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# --- Fungsi Preprocessing untuk TextRank (Extractive) ---
def preprocess_single_article_extractive(article_content):
    cleaned_sentences_tokenized = []
    original_relevant_sentences = []
    stop_words = set(stopwords.words('indonesian'))

    sentences_from_content = sent_tokenize(article_content)

    for sentence in sentences_from_content:
        # Hapus placeholder seperti [baca:...] atau advertisement
        cleaned_sent_for_analysis = re.sub(r'\[baca:\s*[^\]]*\]', '', sentence, flags=re.IGNORECASE)
        cleaned_sent_for_analysis = re.sub(r'advertisement', '', cleaned_sent_for_analysis, flags=re.IGNORECASE)

        words = [word.lower() for word in word_tokenize(cleaned_sent_for_analysis)]
        # Filter kata yang hanya berisi huruf dan bukan stop words
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

        if filtered_words:
            cleaned_sentences_tokenized.append(filtered_words)
            original_relevant_sentences.append(sentence)
    return cleaned_sentences_tokenized, original_relevant_sentences

# --- Fungsi untuk mendapatkan vektor kalimat (Extractive) ---
def get_sentence_vector(sentence_tokens, word2vec_model):
    if word2vec_model is None:
        # Mengembalikan array nol jika model tidak dimuat, sesuaikan dimensi jika perlu
        # Asumsi ukuran vektor 100, bisa disesuaikan jika model Word2Vec Anda berbeda
        return np.zeros(100) 
    
    vectors = []
    for word in sentence_tokens:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)
    # Penting: Mengembalikan vektor nol dengan ukuran yang benar jika tidak ada kata yang valid
    return np.zeros(word2vec_model.vector_size)


# --- Fungsi TextRank untuk Peringkasan (Extractive) ---
def textrank_summarize(article_content, word2vec_model, target_word_count):
    # Dapatkan semua kalimat asli untuk fallback dan pemesanan akhir
    all_original_sentences_in_order = sent_tokenize(article_content)
    
    # Fallback summary jika terjadi kegagalan atau teks terlalu pendek
    fallback_summary = " ".join(all_original_sentences_in_order[:min(len(all_original_sentences_in_order), 2)])

    cleaned_sentences_tokenized, original_relevant_sentences = preprocess_single_article_extractive(article_content)

    if not cleaned_sentences_tokenized or len(cleaned_sentences_tokenized) < 2:
        return fallback_summary, "Gagal (Teks Sangat Pendek atau Tidak Relevan)"

    sentence_vectors = []
    valid_original_indices = []

    for i, tokens in enumerate(cleaned_sentences_tokenized):
        vec = get_sentence_vector(tokens, word2vec_model)
        # Pastikan vektor tidak nol dan memiliki dimensi yang benar
        if vec is not None and not np.all(vec == 0) and vec.shape[0] == word2vec_model.vector_size: 
            sentence_vectors.append(vec)
            valid_original_indices.append(i) # Indeks di dalam original_relevant_sentences

    if len(sentence_vectors) < 2:
        return fallback_summary, "Gagal (Vektor Kalimat Kurang)"

    try:
        # Menghitung matriks kemiripan kosinus antar vektor kalimat
        similarity_matrix = cosine_similarity(sentence_vectors)
    except ValueError as e:
        return fallback_summary, f"Gagal (Masalah Dimensi Vektor: {e})"

    # Membuat graph dari matriks kemiripan dan menjalankan PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = {}
    try:
        # PageRank untuk menentukan skor pentingnya setiap kalimat
        scores = nx.pagerank(graph, max_iter=1000, tol=1e-3)
    except nx.PowerIterationFailedConvergence:
        st.warning("Peringatan: PageRank gagal konvergen untuk teks ini. Menggunakan fallback.")
        return fallback_summary, "Gagal (PageRank Konvergensi)"

    # Urutkan kalimat berdasarkan skor PageRank dari yang tertinggi
    ranked_processed_indices = sorted(((scores[i], idx_in_valid) for idx_in_valid, i in enumerate(scores)), reverse=True)

    current_word_count = 0
    final_summary_pairs = [] # Untuk menyimpan (indeks_asli, kalimat_asli)
    num_selected_sentences = 0

    # Ambil kalimat hingga mencapai target_word_count atau proporsi tertentu
    for score, processed_idx_in_valid in ranked_processed_indices:
        # Dapatkan indeks kalimat asli dari `original_relevant_sentences`
        original_idx_in_relevant = valid_original_indices[processed_idx_in_valid]
        original_sentence = original_relevant_sentences[original_idx_in_relevant]

        # Temukan posisi kalimat asli di keseluruhan artikel untuk menjaga urutan
        try:
            actual_original_idx = all_original_sentences_in_order.index(original_sentence)

            # Hindari duplikasi kalimat (meskipun jarang dengan TextRank)
            if original_sentence not in [s for _, s in final_summary_pairs]:
                final_summary_pairs.append((actual_original_idx, original_sentence))
                current_word_count += len(word_tokenize(original_sentence))
                num_selected_sentences += 1
        except ValueError:
            # Kalimat mungkin telah dimodifikasi atau tidak ditemukan, lewati saja
            pass

        # Hentikan jika sudah mencapai target kata atau cukup banyak kalimat
        # Pilih sekitar 20-50% dari kalimat asli untuk ringkasan
        if current_word_count >= target_word_count or num_selected_sentences >= len(original_relevant_sentences) * 0.3 + 1:
            break
            
    # Jika tidak ada kalimat yang terpilih, gunakan fallback
    if not final_summary_pairs and len(all_original_sentences_in_order) > 0:
        return fallback_summary, "Gagal (Ringkasan Kosong/Terlalu Pendek)"

    # Urutkan kalimat yang terpilih berdasarkan kemunculannya di teks asli
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

    if len(clean_text.split()) > 512: 
        st.warning("Peringatan: Teks input sangat panjang. Model abstractive mungkin hanya memproses 512 token awal. Untuk hasil terbaik, coba gunakan teks yang lebih ringkas.")

    with torch.no_grad(): 
        input_ids = tokenizer.encode(clean_text, return_tensors='pt', max_length=512, truncation=True)
        
        summary_ids = model.generate(
            input_ids,
            min_length=min_len,
            max_length=max_len,
            num_beams=4,              # Meningkatkan num_beams untuk kualitas yang lebih baik
            repetition_penalty=2.0,   
            length_penalty=1.0,       # Sesuaikan length_penalty
            early_stopping=True,      
            no_repeat_ngram_size=3,   # Mencegah pengulangan n-gram berukuran 3
            use_cache=True,           
            do_sample=True,           
            temperature=0.8,          # Sesuaikan temperature untuk sedikit variasi tapi tetap koheren
            top_k=50,                 
            top_p=0.95                
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, "Sukses"


# --- Pemuatan Model (dengan st.cache_resource) ---
# Word2Vec untuk TextRank
@st.cache_resource
def load_word2vec_model(path=W2V_MODEL_PATH):
    with st.spinner("Memuat model Word2Vec (Extractive)..."):
        if os.path.exists(path):
            try:
                model = Word2Vec.load(path)
                # --- MODIFIKASI DIMULAI DI SINI ---
                # Buat placeholder untuk pesan
                success_message_placeholder = st.empty()
                success_message_placeholder.success("Model Word2Vec berhasil dimuat!")
                time.sleep(3) # Tunggu 3 detik
                success_message_placeholder.empty() # Hapus pesan
                return model
            except Exception as e:
                st.error(f"Gagal memuat model Word2Vec: {e}. Pastikan file '{path}' ada dan tidak rusak.")
                st.info("Anda mungkin perlu melatih dan menyimpan model Word2Vec terlebih dahulu dari korpus teks bahasa Indonesia.")
                return None
        else:
            st.error(f"File model Word2Vec '{path}' tidak ditemukan.")
            st.info("Harap pastikan Anda telah menempatkan file ini di folder yang sama dengan aplikasi Streamlit Anda. Jika belum punya, Anda perlu melatihnya atau mengunduh model pra-terlatih.")
            return None

# IndoBART untuk Abstractive
@st.cache_resource
def load_indobart_model(model_id=ABSTRACTIVE_MODEL_ID): # Menggunakan model_id dari Hugging Face
    with st.spinner(f"Mengunduh dan memuat model IndoBART ({model_id})... Ini mungkin memerlukan waktu beberapa saat."):
        try:
            # Langsung memuat dari Hugging Face Hub menggunakan model_id
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval() # Set model ke mode evaluasi
            
            success_message_placeholder = st.empty()
            success_message_placeholder.success("Model IndoBART berhasil dimuat!")
            time.sleep(3) # Tunggu 3 detik
            success_message_placeholder.empty() # Hapus pesan
            return tokenizer, model
        except Exception as e:
            st.error(f"Gagal memuat model IndoBART dari Hugging Face Hub: {e}.")
            st.info(f"Pastikan ID model '{model_id}' benar dan dapat diakses. Coba periksa koneksi internet Anda.")
            return None, None

# Panggil fungsi untuk memuat kedua model saat aplikasi dimulai
word2vec_model = load_word2vec_model()
indobart_tokenizer, indobart_model = load_indobart_model() # Panggil tanpa argumen path

st.title("📝 Text Summarizer (Extractive & Abstractive)")
st.markdown("Aplikasi ini memungkinkan Anda memilih antara peringkasan *extractive* (TextRank) dan *abstractive* (IndoBART).")
st.markdown("---")

# Bagian instruksi model di sidebar
st.sidebar.header("📚 Panduan Penyiapan Model")
st.sidebar.markdown("""
Aplikasi ini membutuhkan dua model machine learning:
1.  **Word2Vec** untuk peringkasan Extractive (TextRank).
2.  **IndoBART** (Transformer) untuk peringkasan Abstractive.

---
**Bagaimana Cara Menyiapkannya?**

**Untuk Model Word2Vec (`word2vec_model.bin`):**
Model ini perlu dilatih pada korpus teks bahasa Indonesia yang besar. Jika Anda belum memilikinya, Anda perlu melatihnya atau mengunduh model pra-terlatih jika tersedia.
* Pastikan file `word2vec_model.bin` berada di **direktori yang sama** dengan file aplikasi Streamlit ini.

**Untuk Model IndoBART (Abstractive):**
Model ini akan diunduh secara otomatis dari Hugging Face Hub saat aplikasi pertama kali dijalankan.
* **ID Model:** `SukmaPutra/abstractive_model_artifacts`
* Pastikan aplikasi Streamlit Anda memiliki akses internet untuk mengunduh model ini.
""")
st.sidebar.markdown("---")


# Pilihan sumber input
input_source = st.radio(
    "Pilih sumber teks:",
    ("Masukkan Teks Manual", "Dari URL Artikel"),
    key="input_source_radio"
)

user_input_text = ""
article_url = ""

if input_source == "Masukkan Teks Manual":
    user_input_text = st.text_area(
        "Masukkan teks yang ingin diringkas di sini:",
        height=300,
        placeholder="Contoh: Presiden Susilo Bambang Yudhoyono siang nanti dijadwalkan berpidato di hadapan para prajurit dan perwira TNI di Markas Besar TNI Cilangkap, Jakarta Timur. Pidato Presiden terkait dengan peringatan Hari Ulang Tahun ke-60 TNI. Keterangan ini disampaikan Panglima TNI Marsekal Djoko Suyanto di Jakarta, Kamis (5/10). Yudhoyono rencananya akan tiba di Cilangkap sekitar pukul 14.00 WIB. Selain presiden, purnawirawan TNI serta keluarga prajurit juga akan menghadiri acara yang digelar di Lapangan Udara Cilangkap ini. Menurut Djoko, pidato ini merupakan kegiatan rutin yang dilakukan presiden setiap peringatan hari ulang tahun TNI. Selain itu juga sebagai sarana komunikasi antarpimpinan negara dan militer. (DNP/Tim Liputan 6 SCTV)"
    )
else: # input_source == "Dari URL Artikel"
    article_url = st.text_input(
        "Masukkan URL artikel yang ingin diringkas:",
        placeholder="Contoh: https://www.kompas.com/..., https://www.detik.com/...",
        key="article_url_input"
    )
    if article_url:
        if not is_valid_url(article_url):
            st.warning("URL yang Anda masukkan tidak valid. Pastikan format URL benar (misalnya, dimulai dengan `http://` atau `https://`).")
        else:
            with st.spinner("Mengambil teks dari URL... Ini mungkin memerlukan waktu beberapa saat."):
                extracted_text = get_text_from_url(article_url)
                if extracted_text:
                    st.success("Teks dari URL berhasil diambil!")
                    user_input_text = extracted_text # Set user_input_text dengan teks dari URL
                    
                    # Tampilkan sebagian teks yang diambil di expander
                    preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                    st.expander("Lihat Pratinjau Teks yang Diambil").write(preview_text)
                else:
                    st.error("Gagal mengambil teks dari URL. Pastikan URL mengarah ke artikel yang dapat diproses dan tidak ada masalah jaringan.")
                    user_input_text = "" # Pastikan input kosong jika pengambilan gagal

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
    # Cek apakah ada teks yang valid untuk diringkas
    if not user_input_text.strip():
        if input_source == "Masukkan Teks Manual":
            st.warning("Mohon masukkan teks terlebih dahulu untuk diringkas.")
        else: # Dari URL Artikel
            st.warning("Mohon masukkan URL artikel yang valid dan pastikan teks berhasil diambil. Coba URL lain jika masih gagal.")
    else:
        summary = ""
        status = "Gagal"

        if summary_type == "Extractive (TextRank)":
            if word2vec_model is None:
                st.error("Model Word2Vec tidak dapat dimuat. Proses peringkasan TextRank tidak dapat dilakukan. Lihat sidebar untuk panduan penyiapan model.")
            else:
                with st.spinner("Meringkas teks Anda dengan TextRank..."):
                    summary, status = textrank_summarize(
                        user_input_text,
                        word2vec_model,
                        summary_length_options['extractive_word_count']
                    )
        elif summary_type == "Abstractive (IndoBART)":
            if indobart_tokenizer is None or indobart_model is None:
                st.error("Model IndoBART tidak dapat dimuat. Proses peringkasan Abstractive tidak dapat dilakukan. Lihat sidebar untuk panduan penyiapan model.")
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
st.markdown("Dibuat dengan ❤️ oleh Sukma Apri Ananda Putra - UAS PDM 2025")