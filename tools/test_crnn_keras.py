#coding:utf-8
import os
import numpy as np
import sys
os.chdir('../')
sys.path.append(os.getcwd())
import cv2
from PIL import Image
from crnn.utils import strLabelConverter,resizeNormalize

from crnn.network_keras import keras_crnn as CRNN
from config import LSTMFLAG
import tensorflow as tf
graph = tf.get_default_graph()##解决web.py 相关报错问题

from crnn import keys
from config import ocrModelKeras
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]=""

def crnnSource():
    alphabet = keys.alphabetChinese##中英文模型
    converter = strLabelConverter(alphabet)
    print(len(alphabet)+1)
    model = CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=LSTMFLAG)
    print('\nModel:\n',model)
    model.load_weights(ocrModelKeras)
    return model,converter
##加载模型
model,converter = crnnSource()

def crnnOcr(image):
       """
       crnn模型，ocr识别
       image:PIL.Image.convert("L")
       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       transformer = resizeNormalize((w, 32))
       image = transformer(image)
       image = image.astype(np.float32)
       image = np.array([[image]])
       global graph
       with graph.as_default():
          preds       = model.predict(image)
       preds = preds[0]
       preds = np.argmax(preds,axis=2).reshape((-1,))
       sim_pred  = converter.decode(preds)
       return sim_pred

if __name__=='__main__':
    img_path = './test/test1.jpg'
    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    text = crnnOcr(img.convert('L'))
    print(text)
