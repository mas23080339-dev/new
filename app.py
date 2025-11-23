import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load & xử lý dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # Chuẩn hóa cột Từ khóa 
    df["Từ khóa"] = df["Từ khóa"].fillna("").str.replace(";", " ")

    # Gộp các cột để TF-IDF
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Từ khóa"] + " " +
        df["Thương hiệu"].fillna("")
    )

    return df

df = load_data()

# 2) TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["FullText"])

# 3) Giao diện Streamlit
st.set_page_config(page_title="CBF Product Recommendation", layout="wide")
st.title("Hệ thống gợi ý sản phẩm (CBF)")
st.write("Tìm sản phẩm phù hợp dựa trên mô tả / từ khóa bạn nhập vào.")

user_query = st.text_input("Nhập sản phẩm bạn muốn tìm:")

if user_query:
    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    ranking = scores.argsort()[::-1]

    threshold = 0.1

    if scores[ranking[0]] < threshold:
        st.warning("Không tìm thấy sản phẩm phù hợp.")
    else:
        best_idx = ranking[0]
        st.subheader("Sản phẩm của chúng tôi:")
        st.write(f"**Tên:** {df.loc[best_idx, 'Tên sản phẩm']}")
        st.write(f"**Mô tả:** {df.loc[best_idx, 'Mô tả']}")
        st.write(f"**Giá:** {df.loc[best_idx, 'Giá']}")
        st.write(f"**Rate:** {df.loc[best_idx, 'Rate']}")
        st.write(f"**Similarity:** `{scores[best_idx]:.3f}`")

        st.subheader("Có thể bạn thích sản phẩm này:")
        for idx in ranking[1:6]:
            if scores[idx] < threshold:
                break
            st.write(f"**Tên:** {df.loc[idx, 'Tên sản phẩm']}")    
            st.write(f"Giá: {df.loc[idx, 'Giá']}")
            st.write(f"Rate: {df.loc[idx, 'Rate']}")
            st.write(f"Similarity: `{scores[idx]:.3f}`")
            st.write("---")
