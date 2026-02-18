package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3主体块(DeepSeekV3Block)
 * 
 * ⚠️ 重要架构说明（根据论文 arXiv:2410.xxxxx）：
 * DeepSeek-V3 使用纯 MoE 架构，推理和代码生成能力通过 MoE 专家网络自然涌现，
 * 而不是通过显式的推理/代码模块实现。
 * 
 * 核心组件：
 * 1. Token嵌入层 - 将token ID转换为向量表示
 * 2. Transformer层堆叠（集成MoE） - 进行序列建模，包含任务感知路由
 * 3. 输出投影层 - 生成最终logits
 * 
 * 数据流：
 * token_ids → embedding → transformer_layers(MoE) → output
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3Block extends Module {
    
    private final DeepSeekV3Config config;
    
    // 核心组件
    private DeepSeekV3TokenEmbedding tokenEmbedding;
    private List<DeepSeekV3TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3Block(String name, DeepSeekV3Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件
     */
    private void initializeComponents() {
        // 1. 初始化Token嵌入层
        tokenEmbedding = new DeepSeekV3TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        // 2. 初始化Transformer层堆叠（带MoE）
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            DeepSeekV3TransformerBlock block = new DeepSeekV3TransformerBlock(
                name + "_transformer_" + i, config);
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
     * 前向传播（纯MoE架构）
     * 
     * @param inputs inputs[0]为token ID序列 [batch_size, seq_len]
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
        
        // 2. Transformer层堆叠（带MoE，推理和代码能力自然涌现）
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
     * 带详细输出的前向传播（包含MoE损失和任务类型）
     * 
     * @param tokenIds token ID序列 [batch_size, seq_len]
     * @param taskType 任务类型（可选）
     * @return 详细输出结果
     */
    public DetailedForwardResult forwardWithDetails(Variable tokenIds, TaskType taskType) {
        validateInput(tokenIds);
        
        // 1. Token嵌入
        Variable x = tokenEmbedding.forward(tokenIds);
        
        // 2. Transformer层堆叠（收集MoE损失）
        double totalMoELoss = 0.0;
        for (DeepSeekV3TransformerBlock block : transformerBlocks) {
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
        
        return new DetailedForwardResult(
            logits, 
            avgMoELoss,
            taskType != null ? taskType : TaskType.GENERAL
        );
    }
    
    /**
     * 验证输入的有效性
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
     * 估算激活参数数量
     */
    public long getActiveParameterCount() {
        return config.estimateActiveParameterCount();
    }
    
    /**
     * 打印架构信息
     */
    public void printArchitecture() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 主体块架构");
        System.out.println("=".repeat(80));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(80));
        System.out.printf("Token嵌入层: %s\n", tokenEmbedding.getClass().getSimpleName());
        System.out.printf("Transformer块数量: %d (每块集成MoE)\n", transformerBlocks.size());
        System.out.printf("专家数量: %d专家, Top-%d选择\n", 
            config.getNumExperts(), config.getTopK());
        System.out.printf("架构模式: Pre-LayerNorm + 纯MoE (推理和代码能力自然涌现)\n");
        System.out.printf("估算总参数: %s\n", formatParamCount(getParameterCount()));
        System.out.printf("激活参数: %s (%.2f%%)\n", 
            formatParamCount(getActiveParameterCount()),
            config.getActivationRatio());
        System.out.println("=".repeat(80));
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
     * 详细前向传播结果类（简化版，仅包含MoE信息）
     */
    public static class DetailedForwardResult {
        /** 最终logits输出 */
        public final Variable logits;
        /** 平均MoE负载均衡损失 */
        public final double avgMoELoss;
        /** 任务类型 */
        public final TaskType taskType;
        
        public DetailedForwardResult(Variable logits, double avgMoELoss, TaskType taskType) {
            this.logits = logits;
            this.avgMoELoss = avgMoELoss;
            this.taskType = taskType;
        }
        
        @Override
        public String toString() {
            return String.format("DetailedForwardResult{taskType=%s, MoE损失=%.6f}", 
                taskType, avgMoELoss);
        }
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekV3Config getConfig() {
        return config;
    }
    
    /**
     * 获取Transformer块列表
     */
    public List<DeepSeekV3TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
}
