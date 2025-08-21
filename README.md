# DRL求解TSP问题
尝试使用DRL方法解决TSP问题，问题表述为遍历所有坐标点后返回起点，通常以最短路程或最短时间为优化目标，属于典型NP-hard问题。

## MDP建模
参考经典的强化学习建模方法，应该将问题中的每一步建立为搜寻到下一个目标点，但由于该过程不存在即时奖励，  
故将一步定义为***完成一个图实例***的规划，以规划路线的整体用时或者整体长度的负数作为奖励  

## 模型架构
模型架构为Actor-Critic结构，  

***ACTOR***:  
编码器（encoder）: 通过MHA + FFN + 残差连接 + 层归一化构建  
解码器（decoder）: 通过LSTM + 加性注意力机制 生成序列化预测结果  

***CRITIC***:  
主要生成baseline，用于评估图实例的复杂度，  
计算Advantage = R - Baseline减少方差

## 更新规则

1. 训练Actor:
   1. 获取Actor的预测结果，主要计算log_prob_sum
   2. 计算Actor的loss = -log_prob_sum * Advantage
   3. 优化Actor的参数
2. 训练Critic:
   1. 获取Critic的预测结果，主要计算value
   2. 计算Critic的loss = MSE(value, R)
   3. 优化Critic的参数



