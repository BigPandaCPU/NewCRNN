此文件夹下，存放的是训练好的模型文件
共有四个文件夹：
base：该文件夹存放的模型文件时chineseOcr公布的预训练好的模型文件。（该模型文件对应的字典大小为5530）
      ocr-dense.pth  表示CRNN模型不带lstm训练得到的权重文件
      ocr-dense-keras.h5 是由ocr-dense.pth模型转换而来的
      ocr-lstm.pth   表示完成的CRNN模型训练得到的权重文件
 
cfg:该文件夹存放的crnn模型不带lstm结构的，在darknet框架下的crnn模型文件 
      ocr.cfg

finetune:里面存放的是用自己数据在这些模型上微调的结果

convert:里面存放的是pytorch训练的权重文件，格式由pytroch转keras，keras转darknet后的权重文件

这些模型文件的下载见链接：
