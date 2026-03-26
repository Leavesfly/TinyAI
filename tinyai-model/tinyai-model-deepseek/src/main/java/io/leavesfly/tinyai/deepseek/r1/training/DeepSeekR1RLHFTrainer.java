package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.deepseek.r1.training.dataset.DeepSeekR1Dataset;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3Dataset;
import io.leavesfly.tinyai.deepseek.v3.training.DeepSeekV3RLHFTrainer;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-R1 RLHF 训练器（委托 V3 底座实现）
 * 
 * DeepSeek-R1 基于 V3 底座训练，RLHF（人类反馈强化学习）是 V3 训练流程的标准步骤：
 * V3 训练流程：预训练 → 监督微调(SFT) → 强化学习(RLHF)
 * 
 * 因此，RLHF 的核心算法实现位于 V3 的 {@link DeepSeekV3RLHFTrainer}，
 * 本类作为 R1 侧的适配层，将 R1 的模型和数据集转换为 V3 格式后委托执行。
 * 
 * R1 在 V3 RLHF 基础上，进一步通过 RLVR（可验证奖励强化学习）实现推理增强，
 * 这是 R1 独有的训练步骤，参见 {@link DeepSeekR1RLVRTrainer}。
 * 
 * 算法：Reward-weighted Regression（奖励加权回归）
 * L = -reward * sum(mask * CE_loss) / sum(mask)
 * 
 * @author leavesfly
 * @version 3.0
 * @see DeepSeekV3RLHFTrainer
 */
public class DeepSeekR1RLHFTrainer {
    
    private final DeepSeekR1Model r1Model;
    private final DeepSeekR1Dataset r1Dataset;
    private final DeepSeekV3RLHFTrainer v3RlhfTrainer;
    
    private int maxEpochs = 3;
    private float learningRate = 1e-5f;
    private float rewardWeight = 1.0f;
    private float qualityWeight = 0.5f;
    
    /**
     * 构造函数
     * 
     * 将 R1 模型和数据集适配为 V3 格式，委托 V3 的 RLHF Trainer 执行训练。
     * 
     * @param model DeepSeek-R1 模型（基于 V3 底座）
     * @param dataset R1 RLHF 数据集（包含奖励标注）
     */
    public DeepSeekR1RLHFTrainer(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
        this.r1Model = model;
        this.r1Dataset = dataset;
        
        // 将 R1 模型适配为 V3 模型（R1 基于 V3 底座，共享核心架构）
        DeepSeekV3Model v3Model = adaptR1ModelToV3(model);
        
        // 将 R1 数据集适配为 V3 数据集格式
        DeepSeekV3Dataset v3Dataset = adaptR1DatasetToV3(dataset);
        
        // 创建 V3 RLHF 训练器
        this.v3RlhfTrainer = new DeepSeekV3RLHFTrainer(v3Model, v3Dataset);
    }
    
    /**
     * 配置训练参数
     * 
     * @param maxEpochs 最大训练轮数
     * @param learningRate 学习率
     * @param rewardWeight 人类奖励权重
     * @param qualityWeight MoE 质量奖励权重
     * @return 训练器自身（支持链式调用）
     */
    public DeepSeekR1RLHFTrainer configure(int maxEpochs, float learningRate,
                                           float rewardWeight, float qualityWeight) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.rewardWeight = rewardWeight;
        this.qualityWeight = qualityWeight;
        this.v3RlhfTrainer.configure(maxEpochs, learningRate, rewardWeight, qualityWeight);
        return this;
    }
    
    /**
     * 开始 RLHF 训练
     * 
     * 委托 V3 的 {@link DeepSeekV3RLHFTrainer} 执行核心训练逻辑。
     */
    public void train() {
        System.out.println("📌 R1 RLHF 训练委托 V3 底座执行（R1 基于 V3 底座）");
        v3RlhfTrainer.train();
    }
    
    /**
     * 将 R1 模型适配为 V3 模型
     * 
     * R1 基于 V3 底座，核心架构相同（MoE + Transformer），
     * 通过创建等价配置的 V3 模型来复用 RLHF 训练逻辑。
     */
    private DeepSeekV3Model adaptR1ModelToV3(DeepSeekR1Model r1Model) {
        DeepSeekV3Config v3Config = DeepSeekV3Config.createTinyConfig();
        v3Config.setVocabSize(r1Model.getConfig().getVocabSize());
        v3Config.setNLayer(r1Model.getConfig().getNLayer());
        v3Config.setNEmbd(r1Model.getConfig().getNEmbd());
        v3Config.setNHead(r1Model.getConfig().getNHead());
        v3Config.setNumExperts(r1Model.getConfig().getNumExperts());
        v3Config.setTopK(r1Model.getConfig().getTopK());
        return new DeepSeekV3Model("deepseek-r1-as-v3-rlhf", v3Config);
    }
    
    /**
     * 将 R1 数据集适配为 V3 数据集格式
     * 
     * R1 Dataset 包含 sequences、rewards、lossMasks，
     * 转换为 V3 Dataset 的 RLHF 构造函数格式。
     */
    private DeepSeekV3Dataset adaptR1DatasetToV3(DeepSeekR1Dataset r1Dataset) {
        List<int[]> sequences = r1Dataset.getSequences();
        int maxSeqLength = r1Dataset.getMaxSeqLength();
        int batchSize = r1Dataset.getBatchSize();
        
        // 构建任务类型列表（R1 默认为 GENERAL 任务）
        List<TaskType> taskTypes = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            taskTypes.add(TaskType.GENERAL);
        }
        
        // 获取 loss masks 和 rewards
        List<float[]> lossMasks = r1Dataset.getLossMasks();
        List<Float> rewards = r1Dataset.getRewardsList();
        
        // 添加长度一致性校验
        if (!lossMasks.isEmpty() && !rewards.isEmpty()) {
            if (lossMasks.size() != sequences.size() || rewards.size() != sequences.size()) {
                throw new IllegalArgumentException(
                    String.format("数据集长度不一致: sequences=%d, lossMasks=%d, rewards=%d",
                        sequences.size(), lossMasks.size(), rewards.size()));
            }
            // RLHF 模式：带 loss mask + 奖励
            return new DeepSeekV3Dataset(sequences, taskTypes, lossMasks, rewards,
                                         maxSeqLength, batchSize, true);
        } else if (!lossMasks.isEmpty()) {
            if (lossMasks.size() != sequences.size()) {
                throw new IllegalArgumentException(
                    String.format("数据集长度不一致: sequences=%d, lossMasks=%d",
                        sequences.size(), lossMasks.size()));
            }
            // SFT 模式：仅带 loss mask
            return new DeepSeekV3Dataset(sequences, taskTypes, lossMasks,
                                         maxSeqLength, batchSize, true, true);
        } else {
            // 基础模式
            return new DeepSeekV3Dataset(sequences, taskTypes,
                                         maxSeqLength, batchSize, true);
        }
    }
}
