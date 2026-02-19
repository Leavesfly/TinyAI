package io.leavesfly.tinyai.rl.util;

import java.util.HashMap;
import java.util.Map;

/**
 * 训练统计信息管理类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * 【职责】
 * 统一管理强化学习训练过程中的各类统计指标,包括损失、奖励、探索率等。
 * 提供增量更新和统计查询功能,避免在各个Agent中重复实现统计逻辑。
 * 
 * 【使用场景】
 * - DQN/DoubleDQN: 记录平均损失、缓冲区使用率、epsilon衰减
 * - REINFORCE: 记录回合回报、策略损失、基线损失
 * - 通用: 提供统一的训练监控接口
 * 
 * 【设计优势】
 * - 消除重复代码: 各Agent中相同的统计逻辑抽取到此类
 * - 易于扩展: 新增指标只需添加字段和方法
 * - 统一接口: toMap()方法提供标准化的监控数据输出
 */
public class TrainingStatistics {
    
    /**
     * 总损失值累计
     */
    private float totalLoss;
    
    /**
     * 平均损失值
     */
    private float averageLoss;
    
    /**
     * 损失计算次数
     */
    private int lossCount;
    
    /**
     * 总奖励累计
     */
    private float totalReward;
    
    /**
     * 平均奖励
     */
    private float averageReward;
    
    /**
     * 奖励计算次数(回合数)
     */
    private int rewardCount;
    
    /**
     * 更新次数
     */
    private int updateCount;
    
    /**
     * 构造函数 - 初始化所有统计指标为0
     */
    public TrainingStatistics() {
        this.totalLoss = 0.0f;
        this.averageLoss = 0.0f;
        this.lossCount = 0;
        this.totalReward = 0.0f;
        this.averageReward = 0.0f;
        this.rewardCount = 0;
        this.updateCount = 0;
    }
    
    /**
     * 更新损失统计
     * 
     * 【增量式平均计算】
     * 采用增量式更新避免大数值累加导致的精度问题:
     * avg_new = (total + loss) / (count + 1)
     * 
     * @param loss 当前损失值
     */
    public void updateLoss(float loss) {
        totalLoss += loss;
        lossCount++;
        averageLoss = totalLoss / lossCount;
    }
    
    /**
     * 更新奖励统计
     * 
     * @param reward 当前回合奖励
     */
    public void updateReward(float reward) {
        totalReward += reward;
        rewardCount++;
        averageReward = totalReward / rewardCount;
    }
    
    /**
     * 增加更新计数
     */
    public void incrementUpdate() {
        updateCount++;
    }
    
    /**
     * 获取平均损失
     * 
     * @return 平均损失值
     */
    public float getAverageLoss() {
        return averageLoss;
    }
    
    /**
     * 获取平均奖励
     * 
     * @return 平均奖励值
     */
    public float getAverageReward() {
        return averageReward;
    }
    
    /**
     * 获取总损失
     * 
     * @return 总损失值
     */
    public float getTotalLoss() {
        return totalLoss;
    }
    
    /**
     * 获取总奖励
     * 
     * @return 总奖励值
     */
    public float getTotalReward() {
        return totalReward;
    }
    
    /**
     * 获取损失计算次数
     * 
     * @return 损失计算次数
     */
    public int getLossCount() {
        return lossCount;
    }
    
    /**
     * 获取奖励计算次数(回合数)
     * 
     * @return 回合数
     */
    public int getRewardCount() {
        return rewardCount;
    }
    
    /**
     * 获取更新次数
     * 
     * @return 更新次数
     */
    public int getUpdateCount() {
        return updateCount;
    }
    
    /**
     * 重置所有统计信息
     * 
     * 【使用场景】
     * - 开始新的训练阶段
     * - 评估模式切换到训练模式
     * - 需要清空历史统计重新计数
     */
    public void reset() {
        totalLoss = 0.0f;
        averageLoss = 0.0f;
        lossCount = 0;
        totalReward = 0.0f;
        averageReward = 0.0f;
        rewardCount = 0;
        updateCount = 0;
    }
    
    /**
     * 转换为Map格式
     * 
     * 【标准化输出】
     * 提供统一的监控数据格式,便于:
     * - 日志记录
     * - 可视化展示
     * - 性能分析
     * 
     * @return 统计信息Map
     */
    public Map<String, Object> toMap() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("average_loss", averageLoss);
        stats.put("total_loss", totalLoss);
        stats.put("loss_count", lossCount);
        stats.put("average_reward", averageReward);
        stats.put("total_reward", totalReward);
        stats.put("reward_count", rewardCount);
        stats.put("update_count", updateCount);
        return stats;
    }
    
    /**
     * 字符串表示
     * 
     * @return 格式化的统计信息
     */
    @Override
    public String toString() {
        return String.format("TrainingStatistics{avgLoss=%.4f, avgReward=%.2f, updates=%d, episodes=%d}",
                           averageLoss, averageReward, updateCount, rewardCount);
    }
}
