import os
pwd = os.getcwd()
######################OCR模型######################
##是否启用LSTM crnn模型
##OCR模型是否调用LSTM层
GPU = True
LSTMFLAG = False
##模型选择 True:中英文模型 False:英文模型
ocrFlag = 'torch'##ocr模型 支持 keras  torch版本
chinsesModel = True
ocrModelKeras = os.path.join(pwd,"models","convert/ocr-dense-keras.h5")##keras版本OCR，暂时支持dense
if chinsesModel:
    if LSTMFLAG:
        ocrModel  = os.path.join(pwd,"models","base/ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd,"models","base/ocr-dense.pth")
else:
        ##纯英文模型
        LSTMFLAG=True
        ocrModel = os.path.join(pwd,"models","ocr-english.pth")
######################OCR模型######################
