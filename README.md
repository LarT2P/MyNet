# MyNet

通过分类网络的实现, 学习tensorflow :smile:

## 各个网络

* vgg11

    * train_acc: 0.9909;
    * test_acc_from_testdataset: 0.9119

    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.01,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 500

* vgg13

    * train_acc: 0.9882
    * test_acc_from_testdataset: 0.9288

    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.01,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 300

* resnet
    
    * 论文(error)
        
        * 20: 8.75
        * 32: 7.51
        * 44: 7.17
        * 56: 6.97
        
    * train_acc: 
    * test_acc_from_testdataset: 

    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.1,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 300
    
* densenet

    架构:
    ```
    self.per_block_num = (L - 4)//3 if self.base else (L - 4)//6

    DensetNet40_12(k=12) 40=4+3*12 or 4+6*6:
    
    3x3,channel=16(or 2k in BC),padding=same,x1,conv
    
    3x3,1,x12,conv      -> 1x1,1,3x3,1,x6,conv
    
    1x1,1,x1,conv
    output: 32
    2x2,2,avepooling
    
    3x3,1,x12,conv      -> 1x1,1,3x3,1,x6,conv
    
    1x1,1,x1,conv
    output: 16
    2x2,2,avepooling
    
    3x3,1,x12,conv      -> 1x1,1,3x3,1,x6,conv
    
    output: 8
    global avepooling
    dense, softmax
    ```
    * train_acc: 0.9894
    * test_acc_from_testdataset: 0.9176
    
    * 'net_name': 'DenseNet40_12'
    * 'batch_size': 64
    * 'image_size': (32, 32)
    * 'learning_rate': 0.01
    * 'moment': 0.9
    * 'display_step': 100
    * 'num_epochs': 500
       
* senet
    
    * 论文(error)
        
        * resnet-110-se: 5.21
        * resnet-164-se: 4.39
        
    * train_acc: 0.9588
    * test_acc_from_testdataset: 0.9011
    
    * 'net_name'     : 'SE_Resnet20',
    * 'batch_size'   : 128,
    * 'image_size'   : (32, 32),
    * 'learning_rate': 0.01,
    * 'moment'       : 0.9,
    * 'display_step' : 100,
    * 'num_epochs'   : 300   
    
    
      
---

DneseNet在训练时十分消耗内存，这是由于算法实现不优带来的。当前的深度学习框架对 DenseNet 的密集连接没有很好的支持，所以只能借助于反复的拼接（Concatenation）操作，将之前层的输出与当前层的输出拼接在一起，然后传给下一层。对于大多数框架（如Torch和TensorFlow），每次拼接操作都会开辟新的内存来保存拼接后的特征。这样就导致一个 L 层的网络，要消耗相当于 L(L+1)/2 层网络的内存（第 l 层的输出在内存里被存了 (L-l+1) 份）。

---

## Train and test CIFAR10 with tensorflow

### Accuracy

| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG11](https://arxiv.org/abs/1409.1556)              | 91.35%      |
| [VGG13](https://arxiv.org/abs/1409.1556)          | 93.02%      |
| [VGG16](https://arxiv.org/abs/1409.1556)          | 93.62%      |
| [VGG19](https://arxiv.org/abs/1409.1556)         | 93.75%      |
| [Resnet20](https://arxiv.org/abs/1512.03385)       | 94.43%      |
| [Resnet32](https://arxiv.org/abs/1512.03385)  | 94.73%      |
| [Resnet44](https://arxiv.org/abs/1512.03385)  | 94.82%      |
| [Resnet56](https://arxiv.org/abs/1512.03385)       | 95.04%      |
| [Xception](https://arxiv.org/abs/1610.02357)    | 95.11%      |
| [MobileNet](https://arxiv.org/abs/1704.04861)             | 95.16%      |
| [DensetNet40_12](https://arxiv.org/abs/1608.06993) | 94.24% |
| [DenseNet100_12](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [DenseNet100_24](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [DenseNet100_24](https://arxiv.org/abs/1608.06993)| 95.21%  |
| [ResNext50](https://arxiv.org/abs/1611.05431)| 95.21%  |
| [ResNext101](https://arxiv.org/abs/1611.05431)| 95.21%  |
| [SqueezeNetA](https://arxiv.org/abs/1602.07360)| 95.21%  |
| [SqueezeNetB](https://arxiv.org/abs/1602.07360)| 95.21%  |
| [SE_Resnet_50](https://arxiv.org/abs/1709.01507)| 95.21%  |
| [SE_Resnet_101](https://arxiv.org/abs/1709.01507)| 95.21%  |


### Net implement
- [x] VGG
- [x] ResNet
- [x] DenseNet
- [x] mobileNet
- [x] ResNext
- [x] Xception
- [x] SeNet
- [x] SqueenzeNet 