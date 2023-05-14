# импортируем библиотеку streamlit
import streamlit as st           # version 1.16.0

# импортируем общие библиотеки
import os
import numpy as np               # version 1.24.3
import pandas as pd              # version 1.5.3
import pysrt                     # version 1.1.2
import spacy                     # version 3.3.1
import en_core_web_sm            # version 3.3.0
import re
import chardet                   # version 5.1.0
from joblib import load          # version 1.1.1

# путь загружаемой модели
MODEL_NAME = '../models/model_bayesNB.joblib'

# путь загружаемого словаря oxford
OXFORD_NAME = '../oxford/classic_oxford.joblib'

# путь к директории проекта
PATHNAME = os.path.dirname(__file__)

# загрузка модели
model = load(os.path.join(PATHNAME, MODEL_NAME))

# загрузка словаря oxford
oxford_words = load(os.path.join(PATHNAME, OXFORD_NAME))

# словарь с кодировкой уровня сложности
LEVEL = {1: 'A2',
         2: 'B1',
         3: 'B2',
         4: 'C1'}

# Header страницы
st.set_page_config(page_title='Уровень английского в субтитрах',
                   page_icon='🎞',
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

with st.container():
    st.header('Узнайте уровень английского по субтитрам фильмов 🎞!')

sub_file = st.file_uploader('Загрузите файл с субтитрами в формате .srt', type='.srt', key=None)
st.markdown('Скачать субтитры для фильма вы можете на сайте: https://www.opensubtitles.org/ru/search/sublanguageid-eng')

# зададим регулярные выражения для очистки текста
HTML = r'<.*?>'  # html тэги меняем на пробел
TAG = r'{.*?}'  # тэги меняем на пробел
COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]'  # комменты в скобках меняем на пробел
LETTERS = r'[^a-zA-Z\.,!? ]'  # все что не буквы меняем на пробел
SPACES = r'([ ])\1+'  # повторяющиеся пробелы меняем на один пробел
DOTS = r'[\.]+'  # многоточие меняем на точку


# функция для очистки субтитров
def clean_subs(subs):
    subs = subs[1:]  # удаляем первый рекламный субтитр
    txt = re.sub(HTML, ' ', subs.text)  # html тэги меняем на пробел
    txt = re.sub(COMMENTS, ' ', txt)  # комменты в скобках меняем на пробел
    txt = re.sub(LETTERS, ' ', txt)  # все что не буквы меняем на пробел
    txt = re.sub(DOTS, r'.', txt)  # многоточие меняем на точку
    txt = re.sub(SPACES, r'\1', txt)  # повторяющиеся пробелы меняем на один пробел
    txt = re.sub('www', '', txt)  # кое-где остаётся www, то же меняем на пустую строку
    txt = txt.lstrip()  # обрезка пробелов слева
    txt = txt.encode('ascii', 'ignore').decode()  # удаляем все что не ascii символы
    txt = txt.lower()  # текст в нижний регистр
    return txt


# функция возвращающая количество уникальных лемм
# в тексте субтитров по каждому уровню
def lemma_count(lemmas, oxf, cat):
    func_dict = {'A1': 0,
                 'A2': 1,
                 'B1': 2,
                 'B2': 3,
                 'C1': 4}
    level = func_dict[cat]
    oxf_word_list = oxf[level].split()
    words = [lemma for lemma in lemmas if lemma in oxf_word_list]

    return len(set(words))


# загрузим для фильма субтитры с использованием библиотеки pysrt
def sub_process(subs):
    # вызов функии для очистки текста
    cln_subs = clean_subs(subs)
    df = pd.DataFrame({'subs': cln_subs}, index=[0])
    # используем библиотеку spacy для лемматизации
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(cln_subs)
    lemma_list = [token.lemma_ for token in doc]
    # в цикле по каждой метке запишем в датафрейм кол-во уникальных лемм
    for lvl in ['A1', 'A2', 'B1', 'B2', 'C1']:
        df.loc[0, lvl + '_lemma_cnt'] = lemma_count(lemma_list, oxford_words, lvl)
    return df


# в случае загруженного файла, определяем кодировку и открываем с помощью pysrt
if sub_file:
    with st.spinner('определяем уровень сложности...'):
        try:
            encode = chardet.detect(sub_file.getvalue())['encoding']
            subtitles = pysrt.from_string(sub_file.getvalue().decode(encode))
            df = sub_process(subtitles)
            predictions = model.predict(df)
            st.subheader(f'Уровень сложности для выбранного фильма: {LEVEL[predictions[0]]}')
            # подсчет количества уникальных слов Oxford в тексте субтитров
            with st.container():
                a1 = df['A1_lemma_cnt'].values[0]
                a2 = df['A2_lemma_cnt'].values[0]
                b1 = df['B1_lemma_cnt'].values[0]
                b2 = df['B2_lemma_cnt'].values[0]
                c1 = df['C1_lemma_cnt'].values[0]
                word_all = a1 + a2 + b1 + b2 + c1
                a1_ratio = a1 / word_all
                a2_ratio = a2 / word_all
                b1_ratio = b1 / word_all
                b2_ratio = b2 / word_all
                c1_ratio = c1 / word_all
                st.write('Узнайте, сколько уникальных слов определенного уровня встречаются в фильме согласно классического Oxford словаря:')
                st.progress(a1_ratio, text=f"слов уровня A1: {round(a1)}")
                st.progress(a2_ratio, text=f"слов уровня A2: {round(a2)}")
                st.progress(b1_ratio, text=f"слов уровня B1: {round(b1)}")
                st.progress(b2_ratio, text=f"слов уровня B2: {round(b2)}")
                st.progress(c1_ratio, text=f"слов уровня C1: {round(c1)}")

        except Exception as e:
            st.error(f'Что-то пошло не так. Попробуйте загрузить файл еще раз!')
            print(e)