# Fuck-SJTU-ML-Exam

### 说明

该仓库面向SJTU SE-125 机器学习  的期末考试

这个仓库主要记录了个人根据老师提供的考纲文档 整理的笔记。 其中大部分都是根据文档上网查阅资料并copy的，夹带有一定私货。 仓库中整理的文档与老师上课的slides关系不大（因为我真的看不懂那么多公式推导）。希望这些文档可以给自己和大家都带来一些帮助。

### 更新说明

12.23 创建仓库 ，更新至 14.深度卷积神经网络（未开始）

12.24 基本全部完成，但是强化学习和集成学习感觉很迷。可能整理的也不太好，还需要完善。

### 考纲

提供者：顾小东老师

| 教学内容                       | 要点                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| 机器学习导论                   | 机器学习概念、核心要素、分类及一些重要概念如泛化、过拟合、数据划分等 |
| 数学基础知识回顾               | 频率主义和贝叶斯主义，贝叶斯公式，梯度                       |
| 线性回归                       | 原理和基本公式，基本的矩阵形式和结果                         |
| 概率参数估计                   | 极大似然估计要理解原理，了解常用的共轭分布                   |
| 贝叶斯分类                     | 贝叶斯网络原理，朴素贝叶斯原理                               |
| 线性判别分析                   | 生成式模型和判别式模型、判别式分类器原理                     |
| 支持向量机与Kernel技术         | 理解最基本的SVM的优化目标，kernel基本原理了解                |
| 逻辑回归                       | 理解原理                                                     |
| 多层感知机                     | MLP概念、优点，激活函数，理解误差反向传播                    |
| 参数优化理论                   | 理解几个概念：在线学习和离线学习，SGD                        |
| 应用：基于神经网络的词向量建模 | 知道词向量原理，会在实际场景下想到用word2vec解决问题         |
| 深度循环神经网络               | 深度学习的概念、表征学习、端到端学习，理解为什么深度学习比浅层学习要好。  RNN和LSTM的概念，梯度爆炸和消失，Sequence-to-Sequence Learning的模型结构和原理, Attention的原理,  Transformer的模型结构, BERT原理 |
| 深度卷积神经网络               | 卷积神经网络的工作原理，几个重要概念如pooling, dropout       |
| 无监督                         | K-means原理、算法过程、初始点选取                            |
| 生成式模型                     | 了解GAN，VAE的原理，知道应用场景                             |
| 强化学习                       | 强化学习原理                                                 |
| 集成学习                       | 了解集成学习的原理                                           |

### 参考文档

1. https://mmdeeplearning.readthedocs.io/zh/latest/overview/concept.html

   https://zhuanlan.zhihu.com/p/71952151

   https://zhuanlan.zhihu.com/p/59673364

   https://zhuanlan.zhihu.com/p/33426884

2. https://zhuanlan.zhihu.com/p/84137223

   https://baike.baidu.com/item/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%85%AC%E5%BC%8F

   https://baike.baidu.com/item/%E6%A2%AF%E5%BA%A6/13014729

3. https://zhuanlan.zhihu.com/p/45023349

4. https://baike.baidu.com/item/%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1

   https://zhuanlan.zhihu.com/p/26638720

   https://zhuanlan.zhihu.com/p/103854460

5. https://zhuanlan.zhihu.com/p/157162433

   https://zhuanlan.zhihu.com/p/30139208
   
6. https://www.zhihu.com/question/20446337

    https://baike.baidu.com/item/%E7%BA%BF%E6%80%A7%E5%88%A4%E5%88%AB%E5%88%86%E6%9E%90/22657333

7. https://zhuanlan.zhihu.com/p/49331510

   《机器学习》 周志华

8. https://zhuanlan.zhihu.com/p/74874291

9. https://zhuanlan.zhihu.com/p/23937778

   https://www.jiqizhixin.com/graph/technologies/7332347c-8073-4783-bfc1-1698a6257db3

10. https://blog.csdn.net/weixin_42267615/article/details/102973252

    https://www.jianshu.com/p/37223e45c838

    https://www.jiqizhixin.com/graph/technologies/8e284b12-a865-4915-adda-508a320eefde

    https://zhuanlan.zhihu.com/p/74571263

11. https://www.zhihu.com/question/32275069/answer/109446135

12. https://medium.com/@pkqiang49/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-%E7%99%BD%E8%AF%9D%E8%A7%A3%E9%87%8A-8%E4%B8%AA%E4%BC%98%E7%BC%BA%E7%82%B9-4%E4%B8%AA%E5%85%B8%E5%9E%8B%E7%AE%97%E6%B3%95-2d34c5cb7175

    https://baike.baidu.com/item/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/3729729

    https://baike.baidu.com/item/%E8%A1%A8%E5%BE%81%E5%AD%A6%E4%B9%A0/2140515

    https://www.zhihu.com/question/50454339/answer/257372299

    https://zhuanlan.zhihu.com/p/217618573

    https://zhuanlan.zhihu.com/p/32085405

13. https://www.cnblogs.com/XDU-Lakers/p/10553239.html

    https://www.jianshu.com/p/80436483b13b

    https://zhuanlan.zhihu.com/p/47063917

    https://www.jianshu.com/p/810ca25c4502

14. https://www.jianshu.com/p/1ea2949c0056

    https://zhuanlan.zhihu.com/p/38200980

15. https://www.jianshu.com/p/4f032dccdcef

    https://easyai.tech/ai-definition/unsupervised-learning/

    https://blog.csdn.net/karine_/article/details/49272189

16. https://www.cnblogs.com/yifanrensheng/p/13586468.html

17. https://bdqfork.cn/articles/46

    https://baike.baidu.com/item/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/2971075

    https://www.zhihu.com/question/26408259/answer/123230350

    https://www.jianshu.com/p/9f113adc0c50

18.https://www.cnblogs.com/WayneZeng/p/9290696.html



































Easteregg：https://www.jiqizhixin.com/graph/technologies/24d01e28-ce75-41a6-9cc2-13d921d8816f

https://www.zhihu.com/question/65403482

https://zhuanlan.zhihu.com/p/38200980

   

