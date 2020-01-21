import pickle
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request, json
from sklearn import preprocessing


def abrirModelo(ruta):
    modelo = pickle.load(open(ruta, 'rb'))
    return modelo


def predecirKNN(modelo, X):
    y = modelo.predict(X)
    return pd.DataFrame(y).idxmax(axis=1)


def predecirRN(modelo, X):
    return modelo.predict_classes(X)


def predecirSVM(modelo, X):
    return modelo.predict(X)


def transformarPCA(pca, X):
    return pca.transform(X)


app = Flask(__name__)


@app.route('/modeloRN', methods=['POST'])
def modeloRN():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    result = str(predecirRN(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\RNModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/modeloRNPCA', methods=['POST'])
def modeloRNPCA():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    datos = transformarPCA(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\PCAModel.h5"), datos)

    result = str(predecirRN(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\RNPCAModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/modeloSVM', methods=['POST'])
def modeloSVM():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    result = str(predecirSVM(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\SVMModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/modeloSVMPCA', methods=['POST'])
def modeloSVMPCA():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    datos = transformarPCA(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\PCAModel.h5"), datos)

    result = str(predecirSVM(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\SVMPCAModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/modeloKNN', methods=['POST'])
def modeloKNN():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    result = str(predecirKNN(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\KNNModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


@app.route('/modeloKNNPCA', methods=['POST'])
def modeloKNNPCA():
    content = request.get_json()

    popularity = content['popularity']
    acousticness = content['acousticness']
    danceability = content['danceability']
    duration_ms = content['duration_ms']
    energy = content['energy']
    instrumentalness = content['instrumentalness']
    key = content['key']
    liveness = content['liveness']
    loudness = content['loudness']
    mode = content['mode']
    speechiness = content['speechiness']
    tempo = content['tempo']
    time_signature = content['time_signature']
    valence = content['valence']

    datos = np.array([[popularity, acousticness, danceability, duration_ms, energy, instrumentalness,
                       key, liveness, loudness, mode, speechiness, tempo, time_signature, valence]])

    datos = preprocessing.scale(datos)

    datos = transformarPCA(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\PCAModel.h5"), datos)

    result = str(predecirKNN(abrirModelo(
        "E:\\Documentos JP\\Artificial\\Proyectos Python\\Flask\\KNNPCAModel.h5"), datos)[0])

    d = {"result": result}

    response = app.response_class(
        response=json.dumps(d),
        status=200,
        mimetype='application/json'
    )

    return response


if __name__ == "__main__":
    app.run(threaded=False)
