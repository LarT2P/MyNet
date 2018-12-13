# MyNet

> [原始仓库](https://github.com/yxlijun/cifar-tensorflow), 作者似乎不再管理, 但是代码结构是真心好.

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
    
* squeezenet

    1. baseline: 
       
        * 'net_name'     : 'SqueezeNetA',
        * 'batch_size'   : 128,
        * 'image_size'   : (32, 32),
        * 'learning_rate': 0.01,
        * 'moment'       : 0.9,
        * 'display_step' : 100,
        * 'num_epochs'   : 200,
        * 'predict_step' : 782,
        * 'restore': False
        
        * train: 0.9029
        * test: 0.8465
        * 分析: 从训练集的准确率来看, 周期数目较少, 准确率较低, 可以尝试将周期数调高100个周期
        
    2. +BN
    
        * train: 0.9461
        * test: 0.8473
        * 分析: BN在一定程度上提升了训练集的准确率, 当然, 测试集也有些许提升
    
    3. => 'num_epochs'  : 300
      
        * train: 0.9361
        * test: 0.8514
        * 分析: 相较于前面的1, 2而言, 测试集准确率有所提升
        
    4. +BN, num_epoch=300
    
        * train: 0.9640
        * test: 0.8472
        * 分析: 这里竟然加了BN后导致测试集下降了, 但是训练集反而是更高了, 这里应该是过拟合比较严重了
        
    5. 4 ~~l2_regularizer~~(这里忘了把正则化添加到损失里, 所以也不算是添加了正则化)
    
        * train: 0.9641
        * test: 0.8523
        * 分析: 这里算是与4实际是一致的, 可是居然有着很大的差异, 课件训练的结果还不稳定, 周期还是太少
        
    6. 5+数据进一步增强(亮度, 色度, 饱和度)
    
        * train: 0.9469
        * test: 0.8463
        * 分析: 这里竟然加了BN后导致测试集下降了, 但是训练集反而是更高了, 这里应该是过拟合比较严重了
        
    7. 5 => num_epochs=500
        
        * model: 
        
            * A 0.9762; 0.8538
            * B 0.9834; 0.8608
            * C 0.9872; 0.8705
            * 分析, 从目前来看, B相对A增加了短路, 准确率有0.007的提升, C对B进一步短路, 又有0.01的提升
            
            * C+relu6: 0.9871; 0.8657
            * C+relu6+余弦衰减: 0.9835; 0.8538
            * 分析, 可以看出来, 余弦衰减, 以及relu6对于这里的情况并不适合.
            
            **最高的测试准确率(最后括号里的内容)现在填入作为标准, 之前没有考虑所测内容并不是最好的模型的对应的准确率**
            
            * C+leakyRuLe(0.2): 0.9821; 0.8624(0.8632)
            * C+leakyRule(0.2)+L2_regularizer: 0.9765; 0.8804 **
            * C+relu6+L2_regularizer: 0.9863; 0.8622(0.8627)
            * C+Rule+L2_regularizer: 0.9803; 0.8661 **
            
    8. 7(C+Rule+L2_regularizer) + 标签平滑
        
        * epsilon = 0.1: 0.9262; 0.8654
        * epsilon = 0.01: 0.9746; 0.8728
        
    9. 7(C+leakyRule(0.2)+L2_regularizer) + 标签平滑
        
        * epsilon = 0.01: 0.9662; 0.8787
        * epsilon = 0.1: 0.9732; 0.8797
        
    10. 9(epsilon = 0.1) => num_epoch = 800: 0.9801; 0.8788
    
        
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