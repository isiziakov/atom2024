from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn

import cv2
from numpy.linalg import norm
import numpy as np

import random


def rejection(image):
    if np.average(norm(image, axis=2)) / np.sqrt(3) < 200 and np.average(norm(image, axis=2)) / np.sqrt(3) > 50:
        return True
    else:
        return False


def smoothing(images):
    for i in range(len(images)):
        images[i] = cv2.blur(images[i], (5, 5))
    return images


types = {
    7: 'Прожог',
    6: 'Непровар',
    5: 'Поры',
    4: 'Кратер',
    2: 'Наплыв',
    1: 'Брызги',
    0: 'Шлаковые включения',
    3: 'Подрез',
    8: 'Трещина'
}


def classify_image(image):
    errors = []
    if random.randint(0, 1) == 1:
        return errors
    else:
        for i in range(random.randint(1, 2)):
            errors.append([random.randint(0, 8), random.uniform(0.0001, 0.4999), random.uniform(0.5, 0.9999),
                           random.uniform(0.0001, 0.4999), random.uniform(0.5, 0.9999)])
        return errors


def get_error_type(err):
    return types[err]


def get_instr(type):
    if type.lower() == "брызги":
        return "Удалите брызги с помощью зубила, шабера или при зачистке шлифовальной машинкой"
    if type.lower() == "подрез":
        return "Выполните подварку или наплавьте дополнительный валик-усилитель"
    if type.lower() == "наплыв":
        return "Зачистите угловой шлифовальной машинкой"
    if type.lower() == "кратер":
        return "Зачистите угловой шлифовальной машинкой и заварите"
    if type.lower() == "трещина":
        return "Зачистите до чистого металла угловой шлифовальной машинкой и заново заварите"
    if type.lower() == "поры":
        return "Выбрать угловой шлифовальной машинкой до чистого металла"
    if type.lower() == "непровар":
        return "Выбрать наплавленный металл, разрезать, зачистить, заново собрать и заварить до чистого металла, угловой шлифовальной машинкой и заново заварить "
    if type.lower() == "шлаковые включения":
        return "Выбрать угловой шлифовальной машинкой до чистого металла и подварить"
    if type.lower() == "прожог":
        return "Выбрать угловой шлифовальной машинкой до чистого металла и подварить"
    return "Про такой дефект информации нет"


def recognise_images(files):
    images = []
    results = []
    for f in files:
        images.append(cv2.imdecode(np.fromstring(f.file.read(), np.uint8), cv2.IMREAD_UNCHANGED))
        results.append('')
    for i in range(len(images)):
        if not rejection(images[i]):
            results[i] = "Изображение некачественное"
    images = smoothing(images)
    for i in range(len(images)):
        if results[i] != '':
            continue
        errors = classify_image(images)
        if len(errors) == 0:
            results[i] = "Дефектов на изображении нет"
        else:
            height, width, _ = images[i].shape
            for err in errors:
                results[
                    i] += f"{get_error_type(err[0])}, {int(err[1] * width)}, {int(err[3] * height)}, {int(err[2] * width)}, {int(err[4] * height)}" + "\n"
            results[i] = results[i].replace(r'\n', '\n')
    return results


app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})


@app.get("/test")
def test():
    return '200'


@app.post("/load_image")
def load_image(f: UploadFile = File(...)):
    f = [f]
    return recognise_images(f)


@app.post("/load_images")
def load_images(files: List[UploadFile] = File(...)):
    return recognise_images(files)


@app.post("/instruction")
def instruction(type):
    return get_instr(type)

uvicorn.run(app, host="127.0.0.1", port=8000)
#if __name__ == "__main__":
#    config = uvicorn.Config(app, port=8089)
#    server = uvicorn.Server(config)
#    await server.serve()
