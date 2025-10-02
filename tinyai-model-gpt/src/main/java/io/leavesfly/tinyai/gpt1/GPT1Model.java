package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.List;
import java.util.ArrayList;

/**
 * GPT-1 模型实现
 * 
 * 继承自Model类，提供GPT-1的高级接口和便捷方法
 * 封装了GPT1Block并提供文本生成、训练、推理等功能
 * 
 * 主要功能：
 * 1. 文本生成（自回归生成）
 * 2. 语言建模预测
 * 3. 模型保存和加载
 * 4. 训练支持
 * 5. 模型信息展示
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1Model extends Model {
    
    /** GPT-1核心块 */
    private GPT1Block gpt1Block;
    
    /** GPT-1配置 */
    private GPT1Config config;
    
    /**
     * 构造GPT-1模型
     * 
     * @param name 模型名称
     * @param config GPT-1配置
     */
    public GPT1Model(String name, GPT1Config config) {
        super(name, new GPT1Block(name + "_block", config));
        this.gpt1Block = (GPT1Block) getBlock();
        this.config = config;
        
        // 设置模型描述
        setDescription("GPT-1 (Generative Pre-trained Transformer 1) 语言模型 - " + 
                      "基于Transformer解码器的自回归语言模型");
    }
    
    /**
     * 使用默认配置构造GPT-1模型
     * 
     * @param name 模型名称
     */
    public GPT1Model(String name) {
        this(name, new GPT1Config());
    }
    
    /**
     * 构造小型GPT-1模型（用于测试和演示）
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param maxSequenceLength 最大序列长度
     */
    public GPT1Model(String name, int vocabSize, int maxSequenceLength) {
        this(name, GPT1Config.createTinyConfig(vocabSize, maxSequenceLength));
    }
    
    /**
     * 语言建模前向传播
     * 
     * @param tokenIds 输入token ID序列 shape: (batchSize, sequenceLength)
     * @return 输出logits shape: (batchSize, sequenceLength, vocabSize)
     */
    public Variable predict(Variable tokenIds) {
        return forward(tokenIds);
    }
    
    /**
     * 预测下一个token
     * 
     * @param tokenIds 输入token ID序列
     * @return 下一个token的logits分布
     */
    public Variable predictNextToken(Variable tokenIds) {
        return gpt1Block.predictNextToken(tokenIds);
    }
    
    /**
     * 预测下一个token（使用整数数组输入）
     * 
     * @param tokenIds 输入token ID数组
     * @return 下一个token的logits分布
     */
    public Variable predictNextToken(int[] tokenIds) {
        Variable input = createVariableFromArray(tokenIds);
        return predictNextToken(input);
    }
    
    /**
     * 生成文本序列
     * 
     * @param promptTokens 提示词token序列
     * @param maxLength 最大生成长度
     * @param temperature 温度参数（控制随机性，默认1.0）
     * @return 生成的完整token序列（包含提示词）
     */
    public List<Integer> generateText(List<Integer> promptTokens, int maxLength, double temperature) {
        return gpt1Block.generateSequence(promptTokens, maxLength, temperature);
    }
    
    /**
     * 生成文本序列（使用默认温度）
     * 
     * @param promptTokens 提示词token序列
     * @param maxLength 最大生成长度
     * @return 生成的完整token序列
     */
    public List<Integer> generateText(List<Integer> promptTokens, int maxLength) {
        return generateText(promptTokens, maxLength, 1.0);
    }
    
    /**
     * 生成文本序列（从整数数组开始）
     * 
     * @param promptTokens 提示词token数组
     * @param maxLength 最大生成长度
     * @param temperature 温度参数
     * @return 生成的完整token序列
     */
    public List<Integer> generateText(int[] promptTokens, int maxLength, double temperature) {
        List<Integer> promptList = new ArrayList<>();
        for (int token : promptTokens) {
            promptList.add(token);
        }
        return generateText(promptList, maxLength, temperature);
    }
    
    /**
     * 批量预测（处理多个序列）
     * 
     * @param batchTokenIds 批量token ID序列
     * @return 批量预测结果
     */
    public Variable batchPredict(Variable batchTokenIds) {
        return predict(batchTokenIds);
    }
    
    /**
     * 计算语言建模损失
     * 
     * @param tokenIds 输入token序列
     * @param targetIds 目标token序列（通常是输入序列右移一位）
     * @return 交叉熵损失
     */
    public double computeLanguageModelingLoss(Variable tokenIds, Variable targetIds) {
        Variable logits = predict(tokenIds);
        
        // 实际实现中需要计算交叉熵损失
        // 这里返回一个占位符值
        // TODO: 实现真正的损失计算
        return 0.0;
    }
    
    /**
     * 从整数数组创建Variable
     * 
     * @param tokenIds token ID数组
     * @return Variable对象
     */
    private Variable createVariableFromArray(int[] tokenIds) {
        float[][] inputArray = new float[1][tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            inputArray[0][i] = tokenIds[i];
        }
        return new Variable(NdArray.of(inputArray));
    }
    
    /**
     * 从二维整数数组创建Variable（批量输入）
     * 
     * @param batchTokenIds 批量token ID数组
     * @return Variable对象
     */
    private Variable createVariableFromBatchArray(int[][] batchTokenIds) {
        int batchSize = batchTokenIds.length;
        int sequenceLength = batchTokenIds[0].length;
        
        float[][] inputArray = new float[batchSize][sequenceLength];
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                inputArray[b][s] = batchTokenIds[b][s];
            }
        }
        return new Variable(NdArray.of(inputArray));
    }
    
    /**
     * 验证模型输入
     * 
     * @param tokenIds 输入token序列
     * @throws IllegalArgumentException 如果输入无效
     */
    public void validateInput(Variable tokenIds) {
        gpt1Block.layerForward(tokenIds);  // 这会触发内部验证
    }
    
    /**
     * 检查序列长度是否有效
     * 
     * @param sequenceLength 序列长度
     * @return 是否有效
     */
    public boolean isValidSequenceLength(int sequenceLength) {
        return sequenceLength > 0 && sequenceLength <= config.getMaxSequenceLength();
    }
    
    /**
     * 检查token ID是否有效
     * 
     * @param tokenId token ID
     * @return 是否有效
     */
    public boolean isValidTokenId(int tokenId) {
        return tokenId >= 0 && tokenId < config.getVocabSize();
    }
    
    /**
     * 获取模型容量信息
     * 
     * @return 模型容量描述
     */
    public String getModelCapacity() {
        long params = gpt1Block.getParameterCount();
        double paramsInM = params / 1_000_000.0;
        
        return String.format("参数量: %.2fM (%d)", paramsInM, params);
    }
    
    /**
     * 打印详细的模型信息
     */
    @Override
    public void printModelInfo() {
        System.out.println("=== GPT-1 模型详细信息 ===");
        System.out.println("模型名称: " + getName());
        System.out.println("模型类型: GPT-1 (Generative Pre-trained Transformer 1)");
        System.out.println();
        
        // 架构信息
        System.out.println("--- 架构配置 ---");
        System.out.println("词汇表大小: " + config.getVocabSize());
        System.out.println("最大序列长度: " + config.getMaxSequenceLength());
        System.out.println("隐藏层维度: " + config.getHiddenSize());
        System.out.println("Transformer层数: " + config.getNumLayers());
        System.out.println("注意力头数: " + config.getNumAttentionHeads());
        System.out.println("前馈网络维度: " + config.getIntermediateSize());
        System.out.println("激活函数: " + config.getActivationFunction());
        System.out.println();
        
        // 训练参数
        System.out.println("--- 训练配置 ---");
        System.out.println("残差dropout: " + config.getResidualDropoutProb());
        System.out.println("嵌入dropout: " + config.getEmbeddingDropoutProb());
        System.out.println("注意力dropout: " + config.getAttentionDropoutProb());
        System.out.println("层归一化epsilon: " + config.getLayerNormEpsilon());
        System.out.println("初始化范围: " + config.getInitializerRange());
        System.out.println();
        
        // 模型统计
        System.out.println("--- 模型统计 ---");
        System.out.println(getModelCapacity());
        System.out.println("每个注意力头维度: " + config.getAttentionHeadSize());
        System.out.println();
        
        // 调用父类方法显示训练信息
        System.out.println("--- 训练信息 ---");
        if (getModelInfo() != null) {
            System.out.println(getModelInfo().getSummary());
        } else {
            System.out.println("暂无训练信息");
        }
        
        System.out.println("========================");
    }
    
    /**
     * 创建小型演示模型
     * 
     * @param name 模型名称
     * @return 小型GPT-1模型
     */
    public static GPT1Model createTinyModel(String name) {
        return new GPT1Model(name, 1000, 128);
    }
    
    /**
     * 创建中型演示模型
     * 
     * @param name 模型名称
     * @return 中型GPT-1模型
     */
    public static GPT1Model createMediumModel(String name) {
        GPT1Config config = GPT1Config.createMediumConfig(5000, 256);
        return new GPT1Model(name, config);
    }
    
    /**
     * 创建完整的GPT-1模型（原论文配置）
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return 完整GPT-1模型
     */
    public static GPT1Model createFullModel(String name, int vocabSize) {
        GPT1Config config = new GPT1Config(vocabSize, 512, 768, 12, 12);
        return new GPT1Model(name, config);
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取GPT-1配置
     * 
     * @return GPT-1配置
     */
    public GPT1Config getConfig() {
        return config;
    }
    
    /**
     * 获取GPT-1核心块
     * 
     * @return GPT-1核心块
     */
    public GPT1Block getGPT1Block() {
        return gpt1Block;
    }
    
    /**
     * 获取词汇表大小
     * 
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return config.getVocabSize();
    }
    
    /**
     * 获取最大序列长度
     * 
     * @return 最大序列长度
     */
    public int getMaxSequenceLength() {
        return config.getMaxSequenceLength();
    }
    
    /**
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return config.getHiddenSize();
    }
    
    /**
     * 获取Transformer层数
     * 
     * @return Transformer层数
     */
    public int getNumLayers() {
        return config.getNumLayers();
    }
    
    /**
     * 获取注意力头数
     * 
     * @return 注意力头数
     */
    public int getNumAttentionHeads() {
        return config.getNumAttentionHeads();
    }
}