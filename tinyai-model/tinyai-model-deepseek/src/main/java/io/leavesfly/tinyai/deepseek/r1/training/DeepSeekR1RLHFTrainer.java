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
     * 将 R1 模型适配为 V3 模型，并迁移 R1 已训练好的权重
     *
     * R1 基于 V3 底座，核心架构相同（MoE + Transformer），
     * 通过创建等价配置的 V3 模型并**按参数名深拷贝**迁移权重，
     * 确保 RLHF 训练从 R1 已有的检查点继续，而不是从零开始。
     *
     * 参数名映射规则：
     * - R1 参数命名以 "{name}_main" 开头（例如 "deepseek-r1_main_transformer_0_attn.weight"）
     * - V3 参数命名以 "{name}_main" 开头（例如 "deepseek-r1-as-v3-rlhf_main_transformer_0_attn.weight"）
     * - 去掉模型名前缀后，剩余路径（token_embedding / transformer_i / final_ln / output_proj）结构相同
     * - MTP 头是 V3 特有，R1 没有，跳过迁移（由 V3 端随机初始化）
     */
    private DeepSeekV3Model adaptR1ModelToV3(DeepSeekR1Model r1Model) {
        // 1. 构建等价配置的 V3
        DeepSeekV3Config v3Config = DeepSeekV3Config.createTinyConfig();
        v3Config.setVocabSize(r1Model.getConfig().getVocabSize());
        v3Config.setNLayer(r1Model.getConfig().getNLayer());
        v3Config.setNEmbd(r1Model.getConfig().getNEmbd());
        v3Config.setNHead(r1Model.getConfig().getNHead());
        v3Config.setNumExperts(r1Model.getConfig().getNumExperts());
        v3Config.setTopK(r1Model.getConfig().getTopK());
        DeepSeekV3Model v3Model = new DeepSeekV3Model("deepseek-r1-as-v3-rlhf", v3Config);

        // 2. 按参数相对路径（去掉模型名前缀）做深拷贝迁移
        int matched = 0;
        int total = 0;
        java.util.Map<String, io.leavesfly.tinyai.nnet.core.Parameter> r1Params =
                r1Model.getModule().namedParameters("", true);
        java.util.Map<String, io.leavesfly.tinyai.nnet.core.Parameter> v3Params =
                v3Model.getModule().namedParameters("", true);

        // 构建 R1 的 "相对路径 → Parameter" 映射（去掉 "{r1_name}_main" 前缀）
        String r1Prefix = r1Model.getName() + "_main.";
        java.util.Map<String, io.leavesfly.tinyai.nnet.core.Parameter> r1Relative = new java.util.HashMap<>();
        for (java.util.Map.Entry<String, io.leavesfly.tinyai.nnet.core.Parameter> e : r1Params.entrySet()) {
            String key = e.getKey();
            String relative = key.startsWith(r1Prefix) ? key.substring(r1Prefix.length()) : key;
            r1Relative.put(relative, e.getValue());
        }

        String v3Prefix = v3Model.getName() + "_main.";
        for (java.util.Map.Entry<String, io.leavesfly.tinyai.nnet.core.Parameter> e : v3Params.entrySet()) {
            total++;
            String key = e.getKey();
            String relative = key.startsWith(v3Prefix) ? key.substring(v3Prefix.length()) : key;

            io.leavesfly.tinyai.nnet.core.Parameter src = r1Relative.get(relative);
            if (src != null && src.getValue() != null) {
                io.leavesfly.tinyai.ndarr.NdArray srcValue = src.getValue();
                io.leavesfly.tinyai.ndarr.NdArray dstValue = e.getValue().getValue();
                // 形状不一致时跳过，避免静默错误
                if (dstValue != null && srcValue.getShape().equals(dstValue.getShape())) {
                    e.getValue().setData(srcValue.copy());
                    matched++;
                }
            }
        }

        System.out.printf("📌 R1 → V3 参数迁移完成: %d / %d 个参数已复用 R1 权重（MTP 头等 V3 特有参数保持随机初始化）%n",
                matched, total);
        return v3Model;
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
