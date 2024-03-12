# General Traffic Tasks Framework

## TODO
### 大数据集下的预测任务
数据集可以叫做：large scale long time-series traffic dataset (LSLTD)
此任务数据集比之前大得多，因此，如何利用很重要，借鉴最近的时序模型，要做出如下尝试：
* 用一个更好地transformer（之前研究的st同步）
* 加入series decomposition（autoformer）
* 时序patch化，看看是否要更改历史序列长度
* 去掉图模型，直接应用时序（这个需要解释）；如果不去掉，可以也patch化（参考patchTST note的解释）
* 做一个真正的feature-independent

## Introduction
1. 实现了一个交通时空预测任务的pipeline，包括但不限于下列任务：
    * traffic prediction
    * transfer learning (model update, transfer among cities)
    * imputation (spatial, temporal, random, long-term)

2. 实现多个任务下不同baseline并保存结果
3. 加入一个长时(超过当前所有数据集的序列长度)大规模节点(将常见的数据集节点规模提升一个数量级)数据集
4. 在多个任务上实现不同模型或框架并取得较好结果

## Structure
### Relations among tasks 23.3.17
预测任务:
* 数据集只有一个: x,y的区别仅在time lag上
* 数据预处理: 正常处理。后续可以加上，对不同节点做不同的scaler，这个到时候再改
* 模型构建： 不需要pre-train model 直接实现即可
* 模型训练： 这里还没发现特殊性

迁移任务：
* 数据集有两个：如果是fine-tune模式下，训练和验证数据来自同一个数据集，只需要pre-train数据集给出
节点数目即可；如果是完全无监督的模式下，也只需要单独数据集，所以其实可以将其和预测任务合二为一，加几个其他的参数即可
* 数据预处理: scaler是直接用pre-train训练集传递来的还是用当前数据集的需要考虑，也只要一个
* 模型构建：略微复杂，首先需要pre-train models，而且如果需要finetune，那么哪些层要改动也需要加入
之前已经实现了version0，后续需要看其他pipeline的实现
* 模型训练：指标等和预测一致，本质还是预测任务，但是训练的方式以及其他细节也需要参考其他迁移任务

补全任务：
* 数据集有一个
* 数据预处理：这个比较复杂,参考TimesNet里数据的处理，还有LocaleGN
* 模型构建：和预测任务基本一致，到时候看看和数据处理方式匹配即可
* 模型训练：不同的补全子任务可能不一致，需要格外注意

总结：
数据集写成同一个分支；预处理大致分两类，迁移和预测可以合并；构建模型写两个分支；模型训练靠config分开

### main functions
* read_dataset: input dataset config name, output dataloaders
* build_model: input model

## Problems
> 发现用nn.functional.layer_norm 去替代 nn.LayerNorm后效果很差，且收敛很慢，这个是为啥？

因为layer_norm中有一个affine的值，如果是True会做一次线性变换，相当于多了一些参数映射；
而functional中的layer是纯计算，没有仿射变换。
