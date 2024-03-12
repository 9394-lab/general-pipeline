# General Traffic Tasks Framework

## TODO
### 长序列下的预测任务
模型可以叫做：long time-series bridge (LTB)
* 用一个更好地transformer（之前研究的st同步）
* 时序patch化，看看是否要更改历史序列长度
* 去掉图模型，直接应用时序；如果不去掉，可以也patch化（参考patchTST note的解释）

## Introduction
1. 实现了一个交通时空预测任务的pipeline，包括但不限于下列任务：
    * traffic prediction
    * transfer learning (model update, transfer among cities)
    * imputation (spatial, temporal, random, long-term)

2. 实现多个任务下不同baseline并保存结果

## Structure
预测任务:
* 数据集只有一个: x,y的区别仅在time lag上
* 数据预处理: 正常处理。后续可以加上，对不同节点做不同的scaler，这个到时候再改
* 模型构建： 不需要pre-train model 直接实现即可
* 模型训练： 这里还没发现特殊性

总结：
数据集写成同一个分支；预处理大致分两类，迁移和预测可以合并；构建模型写两个分支；模型训练靠config分开

### main functions
* read_dataset: input dataset config name, output dataloaders
* build_model: input model

## Problems
> 发现用nn.functional.layer_norm 去替代 nn.LayerNorm后效果很差，且收敛很慢，这个是为啥？
因为layer_norm中有一个affine的值，如果是True会做一次线性变换，相当于多了一些参数映射；
而functional中的layer是纯计算，没有仿射变换。
