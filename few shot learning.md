



现状：

      目前关于小样本学习（Few-shot Learning）大多实验和论文都是围绕cv方向，分析可能原因有： 

 图片识别相关领域标签数据集较多，早期关于图像领域的算法研究相对集中，可以用于meta-task.
 图片中的相对位置信息可以转向向量学习，适用胶囊网络之类的结构.




       Few-shot Learning 模型大致可分为三类：Mode Based，Metric Based 和 Optimization Based。

       其中 Model Based 方法旨在通过模型结构的设计快速在少量样本上更新参数，直接建立输入 x 和预测值 P 的映射函数；

       Metric Based 方法通过度量 batch 集中的样本和 support 集中样本的距离，借助最近邻的思想完成分类；

       Optimization Based 方法认为普通的梯度下降方法难以在 few-shot 场景下拟合，因此通过调整优化方法来完成小样本分类的任务。

      

      nlp小样本学习相关数据集：

         1. FewRel 数据集  由Han等人在EMNLP 2018提出，是一个小样本关系分类数据集，包含64种关系用

             于训练，16种关系用于验证和20种关系用于测试，每种关系下包含700个样本。

        2. ARSC 数据集  由 Yu 等人在 NAACL 2018 提出，取自亚马逊多领域情感分类数据，该数据集包含 23

           种亚马逊商品的评论数据，对于每一种商品，构建三个二分类任务，将其评论按分数分为 5、4、 2 三档，

           每一档视为一个二分类任务，则产生 23*3=69 个 task，然后取其中 12 个 task（4*3）作为测试集，其余

           57 个 task 作为训练集。

        3. ODIC 数据集来自阿里巴巴对话工厂平台的线上日志，用户会向平台提交多种不同的对话任务，和多种不

           同的意图，但是每种意图只有极少数的标注数据，这形成了一个典型的 Few-shot Learning 任务，该数据

           集包含 216 个意图，其中 159 个用于训练，57 个用于测试。

   

      数据集资源看：https://github.com/tata1661/FSL-Mate    (a collection of resources for few-shot learning (FSL).)

       

主流思路：

       1.Model Based方法：代表Meta Learning，又称为 learning to learn，在 meta training 阶段将数据集分解为不同

                                   的 meta task，去学习类别变化的情况下模型的泛化能力，在 meta testing 阶段，面对全新的类

                                   别，不需要变动已有的模型，就可以完成分类。                               

                                   形式化来说，few-shot 的训练集中包含了很多的类别，每个类别中有多个样本。在训练阶段，

                                   会在训练集中随机抽取 C 个类别， 每个类别 K 个样本（总共 CK 个数据），构建一个 meta-task，

                                   作为模型的支撑集（support set）输入；再从这 C 个类中剩余的 数据中抽取一批（batch）样

                                   本作为模型的预测对象（batch set）。即要求模型从 C*K 个数据中学会如何区分这 C 个类别，

                                   这样 的任务被称为 C-way K-shot 问题。               

                                   训练过程中，每次训练（episode）都会采样得到不同 meta-task，所以总体来看，训练包含了

                                   不同的类别组合，这种机制使得模型学会不同 meta-task 中的共性部分，比如如何提取重要特

                                   征及比较样本相似等，忘掉 meta-task 中 task 相关部分。通过这种学习机制学到的模型，在面

                                   对新的未见过的 meta-task 时，也能较好地进行分类。

        

            2.Metric Based 方法：如果在 Few-shot Learning 的任务中去训练普通的基于 cross-entropy 的神经网络分类器，

                                   那么几乎肯定是会过拟合，因为神经网络分类器中有数以万计的参数需要优化。相反，很多非参

                                   数化的方法（最近邻、K-近邻、Kmeans）是不需要优化参数的，因此可以在 meta-learning 的框

                                   架下构造一种可以端到端训练的 few-shot 分类器。该方法是对样本间距离分布进行建模，使得同

                                    类样本靠近，异类样本远离。

                                    代表网络有：孪生网络（Simese Network）、匹配网络（Match Network）、原型网络（Prototype Network）、

                                    Relation Network、Induction Network




           3.Optimization Based方法:  相关内容较少，代表文章 Model-agnostic meta-learning for fast adaptation of deep networks.




小样本数据增强方法汇总：

           ![image](https://user-images.githubusercontent.com/38546249/141744570-90e20571-97b0-4a0e-9d08-d6c3c68e6cd0.png)


       

           图像领域可以方便地对样本进行变换，如旋转、翻转、裁剪、去色、模糊等等，从而得到对应的增强版本。

然而，由于语言天然的复杂性，很难找到高效的、同时又保留语义不变的数据增强方法。一些显式生成增强样本的方法包括：

回译：利用机器翻译模型，将文本翻译到另一个语言，再翻译回来。
CBERT[12][13][12][13] ：将文本的部分词替换成[MASK]，然后利用BERT去恢复对应的词，生成增强句子。
意译（Paraphrase）：利用训练好的Paraphrase生成模型生成同义句。

然而这些方法一方面不一定能保证语义一致，另一方面每一次数据增强都需要做一次模型Inference，开销会很大。鉴于此，

   我们考虑了在Embedding层隐式生成增强样本的方法，如图4所示：

对抗攻击（Adversarial Attack）：这一方法通过梯度反传生成对抗扰动，将该扰动加到原本的Embedding矩阵上，

         就能得到增强后的样本。由于生成对抗扰动需要梯度反传，因此这一数据增强方法仅适用于有监督训练的场景。

打乱词序（Token Shuffling）：这一方法扰乱输入样本的词序。由于Transformer结构没有“位置”的概念，模型对Token

       位置的感知全靠Embedding中的Position Ids得到。因此在实现上，我们只需要将Position Ids进行Shuffle即可。

裁剪（Cutoff）：又可以进一步分为两种：
Token Cutoff：随机选取Token，将对应Token的Embedding整行置为零。
Feature Cutoff：随机选取Embedding的Feature，将选取的Feature维度整列置为零。
Dropout：Embedding中的每一个元素都以一定概率置为零，与Cutoff不同的是，该方法并没有按行或者按列的约束。

这四种方法均可以方便地通过对Embedding矩阵（或是BERT的Position Encoding）进行修改得到，因此相比显式生成增强文

本的方法更为高效。

较新的思路：

      一个原因是图片相关的识别任务中的位置信息提供了向量学习的思路，

      一个是图片识别的任务中的特征提取具有通用性，提供了通过多任务提取通用特征的思路。

目前调研的内容有：

Induction Networks for Few-Shot Text Classification

Learning a Universal Template for Few-shot Dataset Generalization




   

1.Induction Networks for Few-Shot Text Classification[https://zhuanlan.zhihu.com/p/95076823]

 该文总结： 利用胶囊网络（标量学习→向量学习，记录图片中的相对信息），通过学习sample所

属于的类别的表示得到class-wise的向量，然后跟输入的query进行对比。适用于图片中包含相对位置信息的学习。

2.Learning a Universal Template for Few-shot Dataset Generalization

该文总结：作者考虑的是多域⼩样本分类问题，即不可⻅的类与样本来⾃于不同的数据源。这种

多域设置下，⼀个关键的问题是如何整合来⾃不同训练域的特征表示。作者提出了通⽤表征Transformer（URT）层，

利⽤元学习动态地重新加权和组合最合适的域特定特征，从⽽利⽤通⽤特征进⾏⼩样本分类。

我们给出了在Meta-Dataset和⼀系列实验的结果以及注意⼒机制的可视化。同样的相关的测试都

是图片相关的数据集，思路可以参考，但是无开源项目参考。




文本相似度总结： https://leovan.me/cn/2020/10/text-similarity/







离题检测计划采用思路：

    设计网络和这个基本一样，设计完之后发现已有资料[4]，基于对比学习的句子表示迁移框架
![image](https://user-images.githubusercontent.com/38546249/141744463-648f0fe9-352e-4ff7-b2a7-1d0a08b70fa1.png)

一个数据增强模块（详见后文），作用于Embedding层，为同一个句子生成两个不同的增强版本（View）。
一个共享的BERT编码器，为输入的句子生成句向量。
一个对比损失层，用于在一个Batch的样本中计算对比损失，其思想是最大化同一个样本不同增强版本句向量的相似度，同时使得不同样本的句向量相互远离。




参考：

【1】[小样本学习（Few-shot Learning）综述]: https://zhuanlan.zhihu.com/p/61215293

【2】[Meta Network]: https://arxiv.org/pdf/1703.00837.pdf)  

【3】[Meta-Learning in Neural Networks: A Survey](https://arxiv.org/pdf/2004.05439.pdf)

【4】[ACL 2021｜美团提出基于对比学习的文本表示模型，效果相比BERT-flow提升8%]（https://tech.meituan.com/2021/06/03/acl-2021-consert-bert.html）

【5】小样本学习及其在美团场景中的应用


