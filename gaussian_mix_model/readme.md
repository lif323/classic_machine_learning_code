# 运行环境
- python 3.8
- numpy 1.19.1
# 实现效果
随机生成了3个高斯分布的数据,设定高斯混合模型的初始化参数,然后使用EM算法对参数进行优化,
通过实验发现:
1. gmm的初始值对模型的拟合效果有极大影响.
2. 对EM算法的理解:
	- E-step 获取 隐变量的后验分布
	- M-step 依据 隐变量后验分布优化模型参数
# 参考
1. http://sofasofa.io/tutorials/gmm_em/
2. https://www.bilibili.com/video/BV13b411w7Xj
