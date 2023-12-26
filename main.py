#=======================================import library============================
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
#=========================================end of library===========================



#===================================Modeling=======================================
pd.set_option("display.max_columns", None)
nltk.download('punkt')
nltk.download('stopwords')

#loading data set
data_film = pd.read_csv("Dataset/movies.csv")

#data cleansing
data_film.fillna(" ", axis=0, inplace=True)

#feature selection
data_film["konten_film"] = data_film["genres"] + " " + data_film["overview"] + " " + data_film["keywords"] + " " + data_film["tagline"] + " " + data_film["cast"] + " " + data_film["director"]

#preprocesing
stop_words = nltk.corpus.stopwords.words('english')

#function normalize document
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

normalisasi_corpus = np.vectorize(normalize_document)
hasil_normalize_corpus = normalisasi_corpus(list(data_film['konten_film']))

#extration feture
vectorizer = TfidfVectorizer()
fitur_vectors = vectorizer.fit_transform(hasil_normalize_corpus)

#cosine score
kemiripan = cosine_similarity(fitur_vectors)
#==================================End of Model=====================================


#==================================Funct Rekomendasi Film===========================
def rekomendasi_film_genre(input_genre):
    # Membuat kondisi yang akan digunakan dalam operasi pencocokan genre
    condition = data_film['genres'].apply(lambda x: any(genre in x for genre in input_genre))

    # Menggunakan kondisi untuk melakukan pencocokan genre
    genres = data_film[condition]
    genres = genres.sort_values(by="vote_count", ascending=False)
    genres = genres["title"].values[0]

    input_film_title = genres.lower()
    if input_film_title == "off":
        st.write("Program selesai")
        st.stop()
    list_judul_film_dari_dataset = data_film["title"].tolist()
    pencarian_judul_terdekat_dari_user = difflib.get_close_matches(input_film_title, list_judul_film_dari_dataset)
    judul_paling_mirip = pencarian_judul_terdekat_dari_user[0]
    index_dari_judul_film = data_film[data_film.title == judul_paling_mirip]['index'].values[0]
    kemiripan_skor = list(enumerate(kemiripan[index_dari_judul_film]))
    urutan_kemiripan_film = sorted(kemiripan_skor, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    i = 2
    for film in urutan_kemiripan_film:
        index = film[0]
        judul_dari_index = data_film[data_film.index == index]['title'].values[0]
        if (i < 7):
            st.write(i - 1, '.', judul_dari_index)
            i += 1

    return recommended_movies

def recommend_similar_movies(input_film_title):
    input_film_title = input_film_title.lower()
    if input_film_title == "off":
        st.write("Program selesai")
        st.stop()
    list_judul_film_dari_dataset = data_film["title"].tolist()
    pencarian_judul_terdekat_dari_user = difflib.get_close_matches(input_film_title, list_judul_film_dari_dataset)

    if not pencarian_judul_terdekat_dari_user:
        return ["Tidak ada judul film yang cocok ditemukan untuk " + input_film_title]

    pencarian_judul_terdekat_dari_user_str = pencarian_judul_terdekat_dari_user[0].lower()

    if input_film_title != pencarian_judul_terdekat_dari_user_str:
        return ["Tidak ada judul film yang cocok ditemukan untuk " + input_film_title]

    judul_paling_mirip = pencarian_judul_terdekat_dari_user[0]
    index_dari_judul_film = data_film[data_film.title == judul_paling_mirip]['index'].values[0]
    kemiripan_skor = list(enumerate(kemiripan[index_dari_judul_film]))
    urutan_kemiripan_film = sorted(kemiripan_skor, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    i = 1

    for film in urutan_kemiripan_film:
        index = film[0]
        judul_dari_index = data_film[data_film.index == index]['title'].values[0]
        if (i < 7):
            if (i == 1):
                st.write('Film yang serupa dengan : ', judul_dari_index)
                i += 1
            else:
                st.write(i - 1, '.', judul_dari_index)
                i += 1

    return recommended_movies
def user_input_title(judul_film):
    ct1 = st.container()
    ct2 = st.container()
    ct3 = st.container()
    col11,col12 = st.columns(2)
    col21,col22,col23 = st.columns(3)
    judul_film = judul_film.lower()
    if judul_film == "off":
        st.write("Program selesai")
        st.stop()

    list_judul_film_dari_dataset = data_film["title"].tolist()

    pencarian_judul_terdekat_dari_user = difflib.get_close_matches(judul_film, list_judul_film_dari_dataset)

    pencarian_judul_terdekat_dari_user_str = "".join(pencarian_judul_terdekat_dari_user[0])

    pencarian_judul_terdekat_dari_user_str = pencarian_judul_terdekat_dari_user_str.lower()

    if judul_film != pencarian_judul_terdekat_dari_user_str:
        with ct1:
            yesNo = st.text_input("Maksud anda film: " + pencarian_judul_terdekat_dari_user_str + "? (yes/no)")
            yesNo = yesNo.lower()
        if yesNo == "yes":
            judul_paling_mirip = pencarian_judul_terdekat_dari_user[0]
            index_dari_judul_film = data_film[data_film.title == judul_paling_mirip]['index'].values[0]
            kemiripan_skor = list(enumerate(kemiripan[index_dari_judul_film]))
            urutan_kemiripan_film = sorted(kemiripan_skor, key=lambda x: x[1], reverse=True)
            i = 1
            for film in urutan_kemiripan_film:
                index = film[0]
                judul_dari_index = data_film[data_film.index == index]['title'].values[0]
                overview_film = data_film[data_film.index == index]['overview'].values[0]
                rating_vote = data_film[data_film.index == index]['vote_average'].values[0]
                film_dirct = data_film[data_film.index == index]['director'].values[0]
                genre_film = data_film[data_film.index == index]["genres"].values[0]
                if (i == 1):
                    with ct2:
                        with col11:
                             st.image("image/pemandangan.png")
                             st.write("Berikut adalah 30 film yang serupa dengan ",judul_dari_index)
                        with col12:
                            st.success('Film anda ditemukan!')
                            st.write('Judul: ', judul_dari_index)
                            st.write("Rating film : ",str(rating_vote),"/10")
                            st.write("Director : ",film_dirct)
                            st.write("Genre : ",genre_film)
                            st.write("Deskripsi film : ",overview_film)
                            i += 1
                else:
                    if (i < 32):
                        if(i<12):
                            with col21:
                                st.write(i - 1, '.', judul_dari_index)
                                i += 1
                        elif (i<22):
                            with col22:
                                st.write(i - 1, '.', judul_dari_index)
                                i += 1
                        elif (i<32):
                            with col23:
                                st.write(i - 1, '.', judul_dari_index)
                                i += 1

        elif yesNo=="no":
            st.write("Tidak ada judul film yang cocok ditemukan untuk :  " + judul_film)
    else:
        judul_paling_mirip = pencarian_judul_terdekat_dari_user[0]
        index_dari_judul_film = data_film[data_film.title == judul_paling_mirip]['index'].values[0]
        kemiripan_skor = list(enumerate(kemiripan[index_dari_judul_film]))
        urutan_kemiripan_film = sorted(kemiripan_skor, key=lambda x: x[1], reverse=True)
        i = 1
        for film in urutan_kemiripan_film:
            index = film[0]
            judul_dari_index = data_film[data_film.index == index]['title'].values[0]
            overview_film = data_film[data_film.index == index]['overview'].values[0]
            rating_vote = data_film[data_film.index == index]['vote_average'].values[0]
            film_dirct = data_film[data_film.index == index]['director'].values[0]
            genre_film = data_film[data_film.index == index]["genres"].values[0]   
            if (i == 1):
                with ct2:
                    with col11:
                        st.image("image/pemandangan.png")
                        st.write("Berikut adalah 30 film yang serupa dengan ",judul_dari_index)
                    with col12:
                        st.success('Film anda ditemukan!')
                        st.write('Judul: ', judul_dari_index)
                        st.write("Rating film : ",str(rating_vote),"/10")
                        st.write("Director : ",film_dirct)
                        st.write("Genre : ",genre_film)
                        st.write("Deskripsi film : ",overview_film)
                        i += 1
            else:
                if (i < 32):
                    if(i<12):
                        with col21:
                            st.write(i - 1, '.', judul_dari_index)
                            i += 1
                    elif (i<22):
                        with col22:
                            st.write(i - 1, '.', judul_dari_index)
                            i += 1
                    elif (i<32):
                        with col23:
                            st.write(i - 1, '.', judul_dari_index)
                            i += 1
def filter_genre(genre):
    selected_genres_series = pd.Series(genre)
    genre_judul_vote = data_film[["title","genres","vote_average"]]
    
    # Membuat kondisi yang akan digunakan dalam operasi pencocokan genre
    condition = genre_judul_vote['genres'].apply(lambda x: any(genre in x for genre in selected_genres_series))
    
    # Menggunakan kondisi untuk melakukan pencocokan genre
    genres = genre_judul_vote[condition]
    genres = genres.sort_values(by="vote_average", ascending=False)
    st.dataframe(genres.reset_index(drop=True), width=3000, height=2700)


#====================================== End Of funct ===============================


#===================================Website=========================================

#===== Setting Interface =========
st.set_page_config(
    page_title="DF : Dunia Film",
    page_icon="🎬",
    layout="centered"
) 
#==== End of seeting interface =========

#==== Loading Css =================

# with open('style.css')as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

#==== End of loading css =========


#=======Layout=============

with st.sidebar:
    st.image('image\movie-icon-15142.png',width=200)
    menuapp = st.radio("MENU",["Cari","Rekomendasi Film"])

#====End of Layout========


    
if menuapp=="Cari":
    with st.container():
        st.header("Ingin Nostalgia dengan film lama? atau ingin nonton film yang update? Udah buruan segera cari di Dunia Film!")
    
    tab1,tab2 = st.tabs(["Judul Film","Genre"])
    with tab1:
        judul_film = st.text_input("Masukan Judulmu disini",key="judul_film_input")
        if judul_film != "":
            user_input_title(judul_film)
        kolom = st.columns(5)
        num = 1
    with tab2:
        genre = st.multiselect(
            'Cari Genre Kesukaan kamu',
            ["War","Action","Mystery","TV","Comedy","Fantasy","Movie","Family","Horror",
             "Foreign","Music","Thriller","Documentary","Drama","Western","Adventure","Fiction",
             "History","Science","Animation","Crime","Romance"])
        
        if genre != "":
            filter_genre(genre)

    
if menuapp=="Rekomendasi Film":
    st.write("Top 5 Rekomendasi film dari Rating")
    kolom = st.columns(5)
    num = 1
    list_top5_by_vote = ["image/stif_upper_lips.webp", "image/me_and_u_5_buck.webp","image\dancer-texas.webp","image/littlebigtop.webp","image/sardaarji.webp"]


    top5_movie = data_film[["title", "vote_average"]]
    top_5_movie_by_average_score = top5_movie.sort_values(by="vote_average", ascending=False)
    top_5_movie_by_average_score = top_5_movie_by_average_score["title"].head().to_list()

    for col_num, (title, image_path) in enumerate(zip(top_5_movie_by_average_score, list_top5_by_vote), start=1):
        with kolom[col_num - 1]:
            st.image(image_path)
            st.write("TOP ",str(num)," : ",title)
            num+=1

    col1,col2,col3 = st.columns(3)
    with col1:
        recommend_similar_movies(top_5_movie_by_average_score[0])
        st.write("Rekomendasi Film Action")
        rekomendasi_film_genre(input_genre="Action")
    with col2:
        recommend_similar_movies(top_5_movie_by_average_score[1])
        st.write("Rekomendasi Film Horror")
        rekomendasi_film_genre(input_genre="Horror")
    with col3:
        recommend_similar_movies(top_5_movie_by_average_score[2])
        st.write("Rekomendasi Film Drama")
        rekomendasi_film_genre(input_genre="Drama")
    

#======================Rekomendasi Sistem=========================================



