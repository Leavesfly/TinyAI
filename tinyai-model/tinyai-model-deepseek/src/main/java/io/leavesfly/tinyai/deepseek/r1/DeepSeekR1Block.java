package io.leavesfly.tinyai.deepseek.r1;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3MoELayer;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3TransformerBlock;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-R1主体块（DeepSeekR1Block）
 * 
 * ⚠️ 重要架构说明（根据论文 arXiv:2501.12948）：
 * DeepSeek-R1 与 DeepSeek-V3 使用完全相同的 MoE 基础架构（671B 参数）
 * 
 * 架构组成：
 * 1. Token嵌入层 - 将token ID转换为向量表示
 * 2. Transformer层堆叠（Pre-LayerNorm + MoE）- 复用 V3 的 TransformerBlock
 * 3. 输出投影层 - 生成最终logits
 * 
 * 数据流：
 * token_ids → embedding → MoE_transformer_layers → final_ln → output_projection
 * 
 * R1 与 V3 的区别：
 * - 架构：完全相同（MoE with 8 experts, Top-2 routing）
 * - 训练：R1 使用纯 RL 训练，V3 使用标准预训练+后训练
 * - 能力：R1 的推理能力通过 RL 训练自然涌现，无需显式推理模块
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekR1Block extends Module {
    
    private final DeepSeekR1Config config;
    
    // 核心组件（复用 V3 的 MoE 架构）
    private DeepSeekR1TokenEmbedding tokenEmbedding;
    private List<DeepSeekV3TransformerBlock> transformerBlocks;  // ✅ 复用 V3 的 MoE TransformerBlock
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config R1配置对象（继承自 DeepSeekBaseConfig）
     */
    public DeepSeekR1Block(String name, DeepSeekR1Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件（使用 V3 的 MoE 架构）
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层
        tokenEmbedding = new DeepSeekR1TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        // 2. 初始化Transformer层堆叠（✅ 复用 V3 的 MoE TransformerBlock）
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            // 将 R1Config 向上转型为 V3Config 使用
            // 因为它们都继承自 DeepSeekBaseConfig，共享相同的 MoE 配置
            io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config v3Config = convertToV3Config(config);
            DeepSeekV3TransformerBlock block = new DeepSeekV3TransformerBlock(
                name + "_transformer_" + i, v3Config);
            transformerBlocks.add(block);
            registerModule("transformer_" + i, block);
        }
        
        // 3. 初始化最终LayerNorm
        finalLayerNorm = new LayerNorm(
            name + "_final_ln",
            config.getNEmbd(),
            (float) config.getLayerNormEpsilon()
        );
        registerModule("final_ln", finalLayerNorm);
        
        // 4. 初始化输出投影层
        outputProjection = new Linear(
            name + "_output_proj",
            config.getNEmbd(),
            config.getVocabSize(),
            false  // 通常不使用偏置
        );
        registerModule("output_proj", outputProjection);
    }
    
    /**
     * 将 R1Config 转换为 V3Config
     * 
     * 因为 R1 和 V3 共享相同的 MoE 基础架构，
     * 我们可以创建一个 V3Config 来复用 V3 的组件
     */
    private io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config convertToV3Config(DeepSeekR1Config r1Config) {
        io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config v3Config = 
            new io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config();
        
        // 复制所有基础配置（从 DeepSeekBaseConfig 继承的）
        v3Config.setVocabSize(r1Config.getVocabSize());
        v3Config.setNPositions(r1Config.getNPositions());
        v3Config.setNEmbd(r1Config.getNEmbd());
        v3Config.setNLayer(r1Config.getNLayer());
        v3Config.setNHead(r1Config.getNHead());
        v3Config.setNInner(r1Config.getNInner());
        v3Config.setNumExperts(r1Config.getNumExperts());
        v3Config.setTopK(r1Config.getTopK());
        v3Config.setExpertHiddenDim(r1Config.getExpertHiddenDim());
        v3Config.setLoadBalanceLossWeight(r1Config.getLoadBalanceLossWeight());
        v3Config.setEnableTaskAwareRouting(r1Config.isEnableTaskAwareRouting());
        v3Config.setNumTaskTypes(r1Config.getNumTaskTypes());
        v3Config.setActivationFunction(r1Config.getActivationFunction());
        v3Config.setResidPdrop(r1Config.getResidPdrop());
        // EmpdPdrop 在 BaseConfig 中是 empdPdrop
        v3Config.setAttnPdrop(r1Config.getAttnPdrop());
        v3Config.setExpertDropout(r1Config.getExpertDropout());
        v3Config.setLayerNormEpsilon(r1Config.getLayerNormEpsilon());
        
        return v3Config;
    }
    
    /**
     * 前向传播（使用 MoE 架构）
     * 
     * @param inputs 输入变量，inputs[0]为token ID序列 [batch_size, seq_len]
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠（使用 V3 的 MoE TransformerBlock）
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            x = block.forward(x);
        }
        
        // 3. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(x);
        
        // 4. 输出投影
        Variable logits = outputProjection.forward(normalized);
        
        return logits;
    }
    
    /**
     * 带任务类型的前向传播
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（用于任务感知路由）
     * @return logits输出 [batch_size, seq_len, vocab_size]
     */
    public Variable forward(Variable tokenIds, TaskType taskType) {
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠（带任务类型）
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            if (taskType != null) {
                // 使用带任务感知路由的前向传播
                DeepSeekV3TransformerBlock.DetailedForwardResult detailedResult = 
                    block.forwardWithDetails(x, taskType);
                x = detailedResult.output;
            } else {
                // 无任务类型时使用普通前向传播
                x = block.forward(x);
            }
        }
        
        // 3. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(x);
        
        // 4. 输出投影
        Variable logits = outputProjection.forward(normalized);
        
        return logits;
    }
    
    /**
     * 带详细输出的前向传播（包含 MoE 损失）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（可选）
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable tokenIds, TaskType taskType) {
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠（收集 MoE 损失）
        double totalMoELoss = 0.0;
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
            // 获取详细结果（包含 MoE 损失）
            DeepSeekV3TransformerBlock.DetailedForwardResult blockResult = 
                block.forwardWithDetails(x, taskType);
            x = blockResult.output;
            totalMoELoss += blockResult.getLoadBalanceLoss();
        }
        double avgMoELoss = totalMoELoss / transformerBlocks.size();
        
        // 3. 最终LayerNorm
        Variable normalized = finalLayerNorm.forward(x);
        
        // 4. 输出投影
        Variable logits = outputProjection.forward(normalized);
        
        return new DetailedForwardResult(logits, avgMoELoss);
    }
    
    /**
     * 验证输入的有效性
     * 
     * @param tokenIds token ID变量
     */
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape())
            );
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions())
            );
        }
    }
    
    /**
     * 估算参数数量
     */
    public long getParameterCount() {
        return config.estimateParameterCount();
    }
    
    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 主体块架构（MoE）");
        System.out.println("=".repeat(70));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(70));
        System.out.printf("Token嵌入层: %s\n", tokenEmbedding.getClass().getSimpleName());
        System.out.printf("Transformer块数量: %d (使用 V3 的 MoE TransformerBlock)\n", transformerBlocks.size());
        System.out.printf("专家数量: %d\n", config.getNumExperts());
        System.out.printf("Top-K选择: %d\n", config.getTopK());
        System.out.printf("架构模式: Pre-LayerNorm + MoE\n");
        System.out.printf("训练方式: 纯RL（推理能力自然涌现）\n");
        System.out.printf("估算总参数: %s\n", formatParamCount(getParameterCount()));
        System.out.printf("激活参数: %s (%.1f%%)\n", 
            formatParamCount(config.estimateActiveParameterCount()),
            config.getActivationRatio());
        System.out.println("=".repeat(70));
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    /**
     * 详细前向传播结果类（MoE 架构）
     */
    public static class DetailedForwardResult {
        /** 最终logits输出 */
        public final Variable logits;
        /** 平均 MoE 负载均衡损失 */
        public final double avgMoELoss;
        
        public DetailedForwardResult(Variable logits, double avgMoELoss) {
            this.logits = logits;
            this.avgMoELoss = avgMoELoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "DetailedForwardResult{logitsShape=%s, avgMoELoss=%.6f}",
                logits.getValue().getShape(), avgMoELoss
            );
        }
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekR1Config getConfig() {
        return config;
    }
    
    /**
     * 获取Transformer块列表（V3 MoE TransformerBlock）
     */
    public List<DeepSeekV3TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
}
