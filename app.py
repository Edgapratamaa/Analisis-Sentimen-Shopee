import streamlit as st
import pandas as pd
import pickle
import time
from collections import Counter
from tensorflow.keras.models import load_model
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---- Set Page Config ----
st.set_page_config(page_title="Analisis Sentimen Ulasan Shopee", layout="wide")

# ---- Custom CSS with Background Image and Styling ----
st.markdown("""
<style>
    body {
        background: url('https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1200&q=80') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Segoe UI', sans-serif;
        color: #333; 
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #673AB7;
    }

    .stButton>button {
        background-color: #673AB7;
        color: white;
        border-radius: 6px;
        width: 100%;
        font-weight: bold;
        padding: 10px;
        transition: transform 0.2s ease;
        animation: none;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        cursor: pointer;
    }

    .stButton>button:active {
        animation: clickEffect 0.3s forwards;
    }

    @keyframes clickEffect {
        0% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(2deg); }
        100% { transform: scale(1) rotate(0deg); }
    }

    .stTextInput input, .stTextArea textarea {
        border: 1px solid #ced4da;
        border-radius: 6px;
        padding: 8px;
    }
    .stSelectbox label, .stFileUploader label {
        color: #673AB7;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center; color:#673AB7;'>📱 Analisis Sentimen Ulasan Aplikasi Shopee</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analisis sentimen ulasan pengguna menggunakan model Deep Learning berbasis LSTM.</p>",
            unsafe_allow_html=True)

# Class pembungkus model dan tokenizer
class ModelContainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None


# Kelas utama aplikasi
class SentimentAnalysisApp:
    def __init__(self):
        self.model = ModelContainer()
        self.df = None

    def load_model(self):
        """Memuat model dan komponen pendukung"""
        try:
            self.model.model = load_model("model_sentimen_lstm.h5")
            with open("tokenizer.pkl", 'rb') as f:
                self.model.tokenizer = pickle.load(f)
            with open("label_encoder.pkl", 'rb') as f:
                self.model.label_encoder = pickle.load(f)
            st.success("✅ Model berhasil dimuat.")
            return True
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return False

    def predict_sentiment(self, text):
        """Contoh prediksi sentimen (mock sementara)"""
        return {
            'text': text,
            'cleaned_text': text.lower(),
            'predicted_sentiment': 'positive' if len(text) % 2 == 0 else 'negative',
            'confidence': 0.75 + (len(text) % 10) * 0.02
        }

    def interactive_demo(self):
        st.markdown("<h3 style='text-align:center;'>💬 Demo Interaktif Analisis Sentimen</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Masukkan ulasan pengguna, lalu klik tombol di bawah untuk menganalisis sentimen.</p>",
                    unsafe_allow_html=True)

        if self.model.model is None:
            st.warning("⚠️ Model belum dimuat. Silakan muat model terlebih dahulu.")
            return

        user_input = st.text_area("Tulis ulasan Anda:", key="demo_input", height=100)

        if st.button("🔍 Analisis Sentimen") and user_input.strip() != "":
            with st.spinner('Menganalisis...'):
                time.sleep(0.5)
                result = self.predict_sentiment(user_input)

                st.markdown("### 🔍 Hasil Analisis:")
                st.markdown(f"**Review:** `{result['text']}`")
                st.markdown(f"**Cleaned Text:** `{result['cleaned_text']}`")

                sentiment = result['predicted_sentiment'].upper()
                confidence = result['confidence']

                col1, col2 = st.columns([3, 1])
                with col1:
                    if sentiment == "POSITIVE":
                        st.markdown(
                            f"<span style='color:green; font-size:20px;'>😊 SENTIMEN: <strong>{sentiment}</strong></span>",
                            unsafe_allow_html=True)
                    elif sentiment == "NEGATIVE":
                        st.markdown(
                            f"<span style='color:red; font-size:20px;'>😞 SENTIMEN: <strong>{sentiment}</strong></span>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<span style='color:orange; font-size:20px;'>😐 SENTIMEN: <strong>{sentiment}</strong></span>",
                            unsafe_allow_html=True)

                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2%}")

                # Progress bar untuk confidence
                st.progress(int(confidence * 100))
                st.markdown("<hr>", unsafe_allow_html=True)

    def generate_report(self):
        if self.df is None:
            st.warning("⚠️ Data belum tersedia untuk membuat laporan.")
            return

        st.subheader("📊 Ringkasan Analisis Sentimen")
        col1, col2 = st.columns(2)

        with col1:
            fig_rating = px.pie(self.df['rating'].value_counts().reset_index(), names='rating', values='count',
                                title="Distribusi Rating Pengguna",
                                color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig_rating, use_container_width=True)

        with col2:
            sentiment_counts = self.df['sentiment_label'].value_counts()
            fig_sentiment = px.bar(sentiment_counts.reset_index(), x='sentiment_label', y='count',
                                   title="Distribusi Sentimen",
                                   color='sentiment_label',
                                   color_discrete_map={'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#FFA726'})
            st.plotly_chart(fig_sentiment, use_container_width=True)

        st.markdown("### 🧠 Kata-Kata Paling Sering Muncul")
        st.markdown("Berikut adalah kata-kata yang sering muncul dari ulasan positif dan negatif.")

        col3, col4 = st.columns(2)

        with col3:
            pos_words = ' '.join(self.df[self.df['sentiment_label'] == 'positive']['cleaned_text'])
            if pos_words.strip():
                wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc_pos, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                st.caption("📈 Kata-kata umum dari ulasan positif.")

        with col4:
            neg_words = ' '.join(self.df[self.df['sentiment_label'] == 'negative']['cleaned_text'])
            if neg_words.strip():
                wc_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc_neg, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                st.caption("📉 Kata-kata umum dari ulasan negatif.")

        st.markdown("### 📄 Laporan Teks")
        report = []
        report.append("=" * 80)
        report.append("SHOPEE APP SENTIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overview
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Reviews Analyzed: {len(self.df)}")
        report.append(f"Average Rating: {self.df['rating'].mean():.2f}/5.0")
        report.append("Rating Distribution:")
        for rating in sorted(self.df['rating'].unique()):
            count = (self.df['rating'] == rating).sum()
            percentage = (count / len(self.df)) * 100
            stars = "⭐" * int(rating)
            report.append(f"  {stars} ({rating}): {count} reviews ({percentage:.1f}%)")

        # Sentiment Analysis
        report.append("\n2. SENTIMENT ANALYSIS RESULTS")
        report.append("-" * 40)
        sentiment_counts = self.df['sentiment_label'].value_counts()
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_counts:
                count = sentiment_counts[sentiment]
                percentage = (count / len(self.df)) * 100
                emoji = "😊" if sentiment == 'positive' else "😐" if sentiment == 'neutral' else "😞"
                report.append(f"  {emoji} {sentiment.capitalize()}: {count} reviews ({percentage:.1f}%)")

        # Key Insights - Bahasa Indonesia
        report.append("\n3. KEY INSIGHTS")
        report.append("-" * 40)
        pos_ratio = (sentiment_counts.get('positive', 0) / len(self.df)) * 100
        neg_ratio = (sentiment_counts.get('negative', 0) / len(self.df)) * 100
        if pos_ratio > 60:
            report.append("Sentimen secara keseluruhan POSITIF - Pengguna umumnya puas.")
        elif neg_ratio > 40:
            report.append("Terjadi tingkat sentimen NEGATIF yang tinggi - Perlu dilakukan evaluasi produk/layanan.")
        else:
            report.append("Sentimen campuran - Harap pantau umpan balik pengguna secara berkala.")
        report.append(f"🎯 Skor Sentimen: {pos_ratio - neg_ratio:.1f}%")

        # Rekomendasi
        report.append("\n4. REKOMENDASI")
        report.append("-" * 40)
        if neg_ratio > 30:
            report.append("📋 Prioritas Tinggi:")
            report.append("  1. Identifikasi pola umum keluhan pengguna.")
            report.append("  2. Perbaiki performa aplikasi dan pengalaman pengguna (UI/UX).")
            report.append("  3. Percepat respons layanan pelanggan.")
        if pos_ratio > 70:
            report.append("🎉 Poin Kuat untuk Dipertahankan:")
            report.append("  1. Pertahankan strategi pengalaman positif pengguna.")
            report.append("  2. Gunakan ulasan positif untuk promosi.")
            report.append("  3. Pantau perubahan sentimen secara berkala.")

        report.append("\n" + "=" * 80)
        report.append("End of Report")
        report.append("=" * 80)

        st.text_area("📄 Laporan Lengkap", value='\n'.join(report), height=400)
        st.download_button(
            label="💾 Download Full Report",
            data='\n'.join(report),
            file_name='shopee_sentiment_report.txt',
            mime='text/plain'
        )
        st.success("✅ Laporan berhasil dibuat.")


def main():
    if 'app' not in st.session_state:
        st.session_state.app = SentimentAnalysisApp()
    app = st.session_state.app

    menu = st.sidebar.selectbox("Navigasi", ["📂 Muat Model", "💬 Demo Interaktif", "📑 Laporan Analisis"])

    if menu == "📂 Muat Model":
        st.header("📂 Muat Model Analisis Sentimen")
        if st.button("Muat Model"):
            app.load_model()

    elif menu == "💬 Demo Interaktif":
        app.interactive_demo()

    elif menu == "📑 Laporan Analisis":
        st.header("📑 Unggah Dataset untuk Analisis")
        uploaded_file = st.file_uploader("Unggah file CSV yang berisi ulasan", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = {'sentiment_label', 'cleaned_text', 'rating'}
                if required_cols.issubset(df.columns):
                    app.df = df
                    st.success("✅ Dataset berhasil dimuat.")
                    app.generate_report()
                else:
                    st.error("❌ File harus memiliki kolom: 'sentiment_label', 'cleaned_text', dan 'rating'.")
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")


if __name__ == "__main__":
    main()