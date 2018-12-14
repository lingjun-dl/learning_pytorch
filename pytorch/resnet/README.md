# pytorch cifar10

This repo is supported by Huawei and SJTU. Many thanks

### training details

- Learning rate

I updated learning rate at every 100 epcoch(you can make it flexible by your own):

![learning rate decay with respect to epochs](https://github.com/lingjun-dl/learning_pytorch/blob/master/pytorch/resnet/imgs/learningrate%20decay.png)

The results are shown as belows:
![loss and acc](https://github.com/lingjun-dl/learning_pytorch/blob/master/pytorch/resnet/imgs/loss%20and%20acc.png)



### To use pretrained model
You can download my model from `Baidu Drive`. Click [here](https://pan.baidu.com/s/1hehR8cs22lVSeA2Pr3cPIQ)

run in command:
```python
python3 main.py -ngpu 1 -train False -dataroot "your data root dir" -download False
```
or 
```python
python3 main.py -ngpu 1 -train False -download True
```

### You can also train it in your own computer

to train with your own parameters:
```python
python3 main.py -ngpu 1 -dataroot yourrootdir -lr 0.01 -epochs 200 -decay 0.98 -outpdir "dir you want to save your training log and model"
```
*Suggest epochs need to be larger than 240*

or you can set other parameters refer to params below

```python
parser.add_argument('-dataset', type=str, required=False, help='cifar10')
parser.add_argument('-dataroot', type=str, required=False, help='path to dataset')
parser.add_argument('-download', type=bool, required=False, help='path to dataset')
parser.add_argument('-workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('-trainBatchSize', type=int, default=64, help='input train batch size')
parser.add_argument('-validBatchSize', type=int, default=128, help='input validation batch size')
parser.add_argument('-testBatchSize', type=int, default=128, help='input test batch size')
parser.add_argument('-imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('-nc', type=int, default=3, help='input image channels')
parser.add_argument('-ncf', type=int, default=64)
parser.add_argument('-epochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('-ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('-adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-lrdecaytype', type=int, default=1, help='learning rate decay:1 for each epoch variation, 2 for epoch range variation')
parser.add_argument('-decay', type=float, default=0.98, help='lr decay for each epoch')
parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('-baseline', type=float, default=0.85, help='baseline for saving trained model')
parser.add_argument('-train', type=bool, default=True, help='Train or test')
parser.add_argument('-outpdir', type=str, default='./ckpt', help='where to save your model')
```

### Best performance: 93.44%
![accuracy](https://github.com/lingjun-dl/learning_pytorch/blob/master/pytorch/resnet/imgs/results.PNG)
