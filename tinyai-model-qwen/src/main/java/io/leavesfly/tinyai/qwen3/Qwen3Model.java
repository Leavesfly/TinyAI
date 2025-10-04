package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.ml.Model;

/**
 * Qwen3模型 - 完整的因果语言模型
 * 
 * 继承自TinyAI的Model类，封装Qwen3Block并提供完整的模型管理功能：
 * 1. 模型初始化和配置管理
 * 2. 前向传播和推理
 * 3. 模型保存和加载
 * 4. 模型信息统计
 * 5. 训练状态管理
 * 
 * 该模型可用于：
 * - 文本生成任务
 * - 语言建模
 * - 对话系统
 * - 文本补全
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Model extends Model {
    
    /** Qwen3配置 */
    private Qwen3Config config;
    
    /** Qwen3核心网络块 */
    private Qwen3Block qwen3Block;
    
    /**
     * 构造Qwen3模型（带语言模型头）
     * 
     * @param name 模型名称
     * @param config Qwen3配置
     */
    public Qwen3Model(String name, Qwen3Config config) {
        this(name, config, true);
    }
    
    /**
     * 构造Qwen3模型
     * 
     * @param name 模型名称
     * @param config Qwen3配置
     * @param withLMHead 是否包含语言模型头
     */
    public Qwen3Model(String name, Qwen3Config config, boolean withLMHead) {
        super(name, createQwen3Block(name + "_block", config, withLMHead));
        
        this.config = config;
        this.qwen3Block = (Qwen3Block) getBlock();
        
        // 初始化模型信息
        initializeModelInfo();
    }
    
    /**
     * 创建Qwen3Block
     * 
     * @param blockName Block名称
     * @param config Qwen3配置
     * @param withLMHead 是否包含语言模型头
     * @return 创建的Qwen3Block
     */
    private static Qwen3Block createQwen3Block(String blockName, Qwen3Config config, boolean withLMHead) {
        // 验证配置
        config.validate();
        
        return new Qwen3Block(blockName, config, withLMHead);
    }
    
    /**
     * 初始化模型信息
     */
    private void initializeModelInfo() {
        if (getModelInfo() != null) {
            // 设置模型架构类型
            getModelInfo().setArchitectureType("Qwen3");
            
            // 设置模型描述
            String description = String.format(
                "Qwen3大语言模型 - %d层解码器，%d注意力头，%d参数",
                config.getNumHiddenLayers(),
                config.getNumAttentionHeads(),
                qwen3Block.countParameters()
            );
            getModelInfo().setDescription(description);
            
            // 添加配置信息作为元数据
            getModelInfo().addMetric("vocab_size", config.getVocabSize());
            getModelInfo().addMetric("hidden_size", config.getHiddenSize());
            getModelInfo().addMetric("num_layers", config.getNumHiddenLayers());
            getModelInfo().addMetric("num_attention_heads", config.getNumAttentionHeads());
            getModelInfo().addMetric("num_key_value_heads", config.getNumKeyValueHeads());
            getModelInfo().addMetric("intermediate_size", config.getIntermediateSize());
            getModelInfo().addMetric("max_position_embeddings", config.getMaxPositionEmbeddings());
        }
    }
    
    /**
     * 创建小型Qwen3模型用于测试
     * 
     * @param name 模型名称
     * @return 小型Qwen3模型
     */
    public static Qwen3Model createSmallModel(String name) {
        Qwen3Config smallConfig = Qwen3Config.createSmallConfig();
        return new Qwen3Model(name, smallConfig);
    }
    
    /**
     * 创建演示Qwen3模型
     * 
     * @param name 模型名称
     * @return 演示Qwen3模型
     */
    public static Qwen3Model createDemoModel(String name) {
        Qwen3Config demoConfig = Qwen3Config.createDemoConfig();
        return new Qwen3Model(name, demoConfig);
    }
    
    /**
     * 获取模型配置
     * 
     * @return Qwen3配置
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取Qwen3核心块
     * 
     * @return Qwen3Block
     */
    public Qwen3Block getQwen3Block() {
        return qwen3Block;
    }
    
    /**
     * 计算模型参数数量
     * 
     * @return 参数总数
     */
    public long countParameters() {
        return qwen3Block.countParameters();
    }
    
    /**
     * 获取模型大小（MB）
     * 
     * @return 模型大小，假设使用FP32
     */
    public double getModelSizeMB() {
        return countParameters() * 4.0 / 1024.0 / 1024.0;
    }
    
    /**
     * 判断模型是否包含语言模型头
     * 
     * @return 是否包含语言模型头
     */
    public boolean hasLMHead() {
        return qwen3Block.isWithLMHead();
    }
    
    /**
     * 获取模型详细信息
     * 
     * @return 详细信息字符串
     */
    @Override
    public String getModelDetailedInfo() {
        StringBuilder info = new StringBuilder();
        info.append("=== Qwen3模型详细信息 ===\n");
        info.append(String.format("模型名称: %s\n", getName()));
        info.append(String.format("架构类型: %s\n", "Qwen3 Transformer"));
        info.append(String.format("参数数量: %,d (%.2f MB)\n", countParameters(), getModelSizeMB()));
        info.append(String.format("包含语言模型头: %s\n", hasLMHead() ? "是" : "否"));
        
        info.append("\n=== 架构配置 ===\n");
        info.append(String.format("词汇表大小: %,d\n", config.getVocabSize()));
        info.append(String.format("隐藏维度: %d\n", config.getHiddenSize()));
        info.append(String.format("中间维度: %d\n", config.getIntermediateSize()));
        info.append(String.format("层数: %d\n", config.getNumHiddenLayers()));
        info.append(String.format("注意力头数: %d\n", config.getNumAttentionHeads()));
        info.append(String.format("键值头数: %d\n", config.getNumKeyValueHeads()));
        info.append(String.format("头维度: %d\n", config.getHeadDim()));
        info.append(String.format("最大位置编码: %d\n", config.getMaxPositionEmbeddings()));
        info.append(String.format("RoPE基础频率: %.1f\n", config.getRopeTheta()));
        info.append(String.format("RMSNorm epsilon: %.0e\n", config.getRmsNormEps()));
        
        info.append("\n=== 特殊标记 ===\n");
        info.append(String.format("填充标记ID: %d\n", config.getPadTokenId()));
        info.append(String.format("开始标记ID: %d\n", config.getBosTokenId()));
        info.append(String.format("结束标记ID: %d\n", config.getEosTokenId()));
        info.append(String.format("共享词嵌入权重: %s\n", config.isTieWordEmbeddings() ? "是" : "否"));
        
        // 添加基础模型信息
        String baseInfo = super.getModelDetailedInfo();
        if (baseInfo != null && !baseInfo.isEmpty()) {
            info.append("\n=== 训练信息 ===\n");
            info.append(baseInfo);
        }
        
        return info.toString();
    }
    
    /**
     * 获取模型简要信息
     * 
     * @return 简要信息字符串
     */
    @Override
    public String getModelSummary() {
        return String.format(
            "Qwen3模型[%s]: %d层, %d头, %,d参数 (%.1fMB)",
            getName(),
            config.getNumHiddenLayers(),
            config.getNumAttentionHeads(),
            countParameters(),
            getModelSizeMB()
        );
    }
    
    /**
     * 验证输入数据
     * 
     * @param inputIds 输入token ID
     * @throws IllegalArgumentException 如果输入无效
     */
    public void validateInput(int[][] inputIds) {
        if (inputIds == null || inputIds.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        for (int[] sequence : inputIds) {
            if (sequence == null || sequence.length == 0) {
                throw new IllegalArgumentException("序列不能为空");
            }
            
            for (int tokenId : sequence) {
                if (tokenId < 0 || tokenId >= config.getVocabSize()) {
                    throw new IllegalArgumentException(
                        String.format("token ID %d 超出词汇表范围 [0, %d)", 
                            tokenId, config.getVocabSize()));
                }
            }
        }
    }
    
    /**
     * 设置模型为推理模式
     * 
     * 该方法主要用于标记模型状态，实际的推理逻辑在TinyAI框架中
     * 通过Variable的计算图自动处理
     */
    public void setInferenceMode() {
        // TinyAI框架中推理模式通过不计算梯度来实现
        // 这里主要用于状态标记和未来扩展  
        if (getModelInfo() != null) {
            getModelInfo().addMetric("inference_mode", 1.0);
        }
    }
    
    /**
     * 设置模型为训练模式
     */
    public void setTrainingMode() {
        if (getModelInfo() != null) {
            getModelInfo().addMetric("inference_mode", 0.0);
        }
    }
}