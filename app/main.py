from flask import Flask, request, Response
from scipy.io import loadmat
import jsonpickle
import numpy
import cv2
app= Flask(__name__)
t1 = loadmat("damoosro01.mat")["Theta1"]
t2 = loadmat("damoosro02.mat")["Theta2"]
def sigmoide(h):
    return 1. / (1 + numpy.e ** (-h))
def frontPropagation(imagen, t1, t2):
    a1 = imagen
    a1 = numpy.append(numpy.array([[1]]), a1, axis=1)
    a2 = t1.dot(a1.T)
    a2 = sigmoide(a2)
    a2 = numpy.append(numpy.array([[1]]), a2, axis=0)
    a3 = t2.dot(a2)
    a3 = sigmoide(a3)
    h = a3
    return h.argmax()
@app.route("/api/test", methods=['POST'])
def test():
    r = request
    nparray = numpy.fromstring(r.data, numpy.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    imagen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(imagen.shape)
    imagen = imagen.T.flatten()  # vector 400 posiciones
    imagen = imagen.reshape(1, imagen.size)
    prediccion = frontPropagation(imagen, t1, t2)
    response = {'message': str(prediccion)}
    print("se predice ", str(prediccion))

    response_pickle = jsonpickle.encode(response)

    return Response(response=response_pickle, status=200, mimetype='application/json')
@app.route('/')
def index():
  return "<h1>Hola more</h1>"