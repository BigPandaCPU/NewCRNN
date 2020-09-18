# NewCRNN
此项目提供了NewCRNN的pytorch训练脚本及将模型文件从pytorch转keras，keras装darknet格式的转换脚本，
另外此模型是在chinseOcr的ocr模型基础上进行整理的

CRNN    = CNN + BiLSTM + CTC_Loss/Beam_Search
NewCRNN = CNN + CTC_Loss/硬解码（Greedy_Search）
提供的转换脚本是为了便于部署，可以直接在OpenCV上部署。
OpenCV支持darknet框架。
训练用pytorch训练，部署基于opencv


pytorch=0.4.0

模型训练:
          
          python  train_ocr.py
          
模型转换脚本：

          bash step1_pytorch_to_keras.sh          
          bash step2_keras_to_darknet.sh
          
致谢：
    
    感谢chineeOcr作者的无私奉献，我在他的基础上做了一点点工作，
    将CRNN权重文件从pytorch转keras，keras转darknet的脚本调试通。
    
补充：模型文件的下载链接

Author:BigPanda
E-mail:wangxiong@founder.com 
State Key Laboratory of Digital Publishing Technology 
Date:2020-9-18
