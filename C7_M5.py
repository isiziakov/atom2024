import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
from st_audiorec import st_audiorec
import cv2
import matplotlib.patches as patches

url = 'http://localhost:8089/'
if 'run' not in st.session_state:
    st.session_state.run = False

if not st.session_state.run:
    st.title("Регистрация")
    fio = st.text_input("ФИО")
    dolzh = st.text_input("Должность")
    if not st.button("Войти") and fio != '' and dolzh != '':
        st.session_state.run = True
else:
    st.write("Регистрация завершена")

    on = st.toggle("Изображдение / аудио")
    if on:
        st.title("Аудио")
        on2 = st.toggle("Выбрать файл / записать")
        if on2:
            wav_audio_data = st_audiorec()
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format='audio/wav')
            st.write("По окончании записи нажмите на кнопку download и отправьте скачанный файл через вкладку загрузки файла")
        else:
            uploaded_file = st.file_uploader("Выберите аудио файл для распознавания", type=['wav'])
            if uploaded_file is not None:
                if st.button("Распознать"):
                    myobj = {'f': uploaded_file}
                    x = requests.post(url + "load_audio", files=myobj)
                    st.write(x.text)

    else:
        st.title("Изображения")
        uploaded_files = st.file_uploader("Выберите аудио файл для распознавания", type=['jpg'], accept_multiple_files=True)
        if uploaded_files is not None:
            st.write("*вероятность правильно определить наличие / дефекта на изображении 0,055")
            if st.button("Распознать"):
                files = [('files', f) for f in uploaded_files]
                myobj = {'files': uploaded_files}
                x = requests.post(url + "load_images", files=files)
                res = x.text.split('","')
                for r in range(len(res)):
                    res[r] = res[r].replace('["','')
                    res[r] = res[r].replace('"]', '')
                    st.write(res[r].replace("\\n", "\n"))
                    image = cv2.imdecode(np.fromstring(uploaded_files[r].getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    fig, ax = plt.subplots()
                    ax.imshow(image)
                    if res[r].find("\\n") > -1:
                        res2 = res[r].split("\\n")
                        for rr in range(len(res2) - 1):
                            s = res2[rr].split(', ')
                            x0 = int(s[1])
                            x1 = int(s[2])
                            y0 = int(s[3])
                            y1 = int(s[4])
                            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1,
                                                    edgecolor='r', facecolor="none")
                            ax.add_patch(rect)
                    st.pyplot(fig)

    st.title("Рекомендации по устранению дефектов")
    error = st.text_input("Название дефекта")
    if st.button("Получить рекомендацию"):
        myobj = {'type': error}
        x = requests.post(url + "instruction", params=myobj)
        st.write(x.text)

    st.title("Дообучение модели работы с аудио")
    df = st.file_uploader("Выберите файл с дообучающим датасетом", type=['csv'])
    if df is not None:
        if st.button("Дообучить"):
            myobj = {'f': df}
            x = requests.post(url + "train", files=myobj)
            st.write(x.text)
