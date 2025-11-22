import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("gucci.csv", encoding="utf-8")
    df["FullText"] = (
        df["Tên sản phẩm"].fillna("") + " " +
        df["Mô tả"].fillna("") + " " +
        df["Thương hiệu"].fillna("")
    )
    return df

df = load_data()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["FullText"])

# UI
st.title("Chúng tôi bán áo cận date :)))")
st.write("Những gì bạn cần chúng tôi không có")

user_input = st.text_input("Nhập tên sản phẩm bạn muốn tìm")

if user_input:
    names = df["Tên sản phẩm"].tolist()
    match = process.extractOne(user_input, names, scorer=fuzz.WRatio)

    if match is None:
        st.error("Không tìm thấy sản phẩm phù hợp!")
    else:
        best_name = match[0]
        index = df[df["Tên sản phẩm"] == best_name].index[0]
        st.success(f"Sản phẩm của chúng tôi đang có: **{best_name}**")

        scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix)[0]
        ranking = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[1:6]

        st.subheader("Gợi ý sản phẩm tương tự:")
        for idx, sc in ranking:
            st.write(f"- **{df.loc[idx, 'Tên sản phẩm']}**  — similarity: `{sc:.3f}`")
