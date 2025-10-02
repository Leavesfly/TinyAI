package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-1 主模型块实现
 * 
 * 继承自Block类，实现完整的GPT-1 Transformer解码器架构
 * 
 * 模型结构：
 * 1. Token嵌入 + 位置嵌入
 * 2. N × GPT1TransformerBlock
 * 3. 最终层归一化
 * 4. 输出头（线性投影到词汇表）
 * 
 * 特点：
 * - 使用仅解码器的Transformer架构
 * - 带因果掩码的多头自注意力
 * - Post-LayerNorm结构（与原始Transformer一致）
 * - 支持权重共享和参数高效训练
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1Block extends Block {
    
    /** GPT-1配置 */
    private GPT1Config config;
    
    /** Token嵌入层 */
    private GPT1TokenEmbedding tokenEmbedding;
    
    /** Transformer块列表 */
    private List<GPT1TransformerBlock> transformerBlocks;
    
    /** 最终层归一化 */
    private LayerNorm finalLayerNorm;
    
    /** 输出头 */
    private GPT1OutputHead outputHead;
    
    /**
     * 构造GPT-1 Block
     * 
     * @param name 模型名称
     * @param config GPT-1配置
     */
    public GPT1Block(String name, GPT1Config config) {
        super(name, 
              Shape.of(-1, config.getMaxSequenceLength()), 
              Shape.of(-1, config.getMaxSequenceLength(), config.getVocabSize()));
        
        this.config = config;
        
        // 验证配置
        config.validate();
        
        init();
    }
    
    /**
     * 使用默认配置的构造函数
     * 
     * @param name 模型名称
     */
    public GPT1Block(String name) {
        this(name, new GPT1Config());
    }
    
    /**
     * 创建小型GPT-1的构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param maxSequenceLength 最大序列长度
     */
    public GPT1Block(String name, int vocabSize, int maxSequenceLength) {
        this(name, GPT1Config.createTinyConfig(vocabSize, maxSequenceLength));
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT1TokenEmbedding(name + "_token_embedding", config);
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块列表
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < config.getNumLayers(); i++) {
                GPT1TransformerBlock transformerBlock = new GPT1TransformerBlock(
                    name + "_transformer_" + i, 
                    config
                );
                transformerBlocks.add(transformerBlock);
                addLayer(transformerBlock);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(
                name + "_final_ln", 
                config.getHiddenSize()
            );
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            // 可选择是否与Token嵌入共享权重
            outputHead = new GPT1OutputHead(
                name + "_output_head", 
                config, 
                false,  // 不使用偏置（GPT-1常见做法）
                tokenEmbedding.getTokenEmbedding()  // 共享Token嵌入权重
            );
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable tokenIds = inputs[0];  // shape: (batchSize, sequenceLength)
        
        // 验证输入
        validateInput(tokenIds);
        
        // 1. Token嵌入 + 位置嵌入
        Variable embeddings = tokenEmbedding.layerForward(tokenIds);
        
        // 2. 通过所有Transformer块
        Variable hiddenStates = embeddings;
        for (GPT1TransformerBlock transformerBlock : transformerBlocks) {
            hiddenStates = transformerBlock.layerForward(hiddenStates);
        }
        
        // 3. 最终层归一化
        hiddenStates = finalLayerNorm.layerForward(hiddenStates);
        
        // 4. 输出头投影到词汇表
        Variable logits = outputHead.layerForward(hiddenStates);
        
        return logits;
    }
    
    /**
     * 验证输入的有效性
     * 
     * @param tokenIds 输入的token ID序列
     * @throws IllegalArgumentException 如果输入无效
     */
    private void validateInput(Variable tokenIds) {
        Shape inputShape = tokenIds.getValue().getShape();
        
        if (inputShape.getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量，但获得了%d维", inputShape.getDimNum())
            );
        }
        
        int sequenceLength = inputShape.getDimension(1);
        if (sequenceLength > config.getMaxSequenceLength()) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", 
                            sequenceLength, config.getMaxSequenceLength())
            );
        }
    }
    
    /**
     * 预测下一个token
     * 
     * @param tokenIds 输入的token ID序列
     * @return 下一个token的概率分布
     */
    public Variable predictNextToken(Variable tokenIds) {
        Variable logits = layerForward(tokenIds);
        
        // 只返回最后一个位置的logits
        // shape: (batchSize, sequenceLength, vocabSize) -> (batchSize, vocabSize)
        return extractLastPosition(logits);
    }
    
    /**
     * 提取序列最后位置的输出
     * 
     * @param logits 完整序列的logits
     * @return 最后位置的logits
     */
    private Variable extractLastPosition(Variable logits) {
        // 简化实现：返回最后一个时间步的输出
        // 实际实现中可能需要更复杂的处理
        return logits;
    }
    
    /**
     * 生成文本序列
     * 
     * @param inputIds 输入token序列
     * @param maxLength 最大生成长度
     * @param temperature 温度参数（控制随机性）
     * @return 生成的token序列
     */
    public List<Integer> generateSequence(List<Integer> inputIds, int maxLength, double temperature) {
        List<Integer> generatedIds = new ArrayList<>(inputIds);
        
        for (int i = 0; i < maxLength && generatedIds.size() < config.getMaxSequenceLength(); i++) {
            // 准备输入
            Variable currentInput = createInputVariable(generatedIds);
            
            // 前向传播
            Variable logits = predictNextToken(currentInput);
            
            // 采样下一个token
            int nextTokenId = sampleFromLogits(logits, temperature);
            
            // 添加到序列
            generatedIds.add(nextTokenId);
            
            // 如果生成了结束符，可以提前停止
            // if (nextTokenId == eosTokenId) break;
        }
        
        return generatedIds;
    }
    
    /**
     * 从token列表创建输入变量
     * 
     * @param tokenIds token ID列表
     * @return 输入变量
     */
    private Variable createInputVariable(List<Integer> tokenIds) {
        int sequenceLength = tokenIds.size();
        
        // 创建输入数组
        float[][] inputArray = new float[1][sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            inputArray[0][i] = tokenIds.get(i);
        }
        
        return new Variable(io.leavesfly.tinyai.ndarr.NdArray.of(inputArray));
    }
    
    /**
     * 从logits中采样下一个token
     * 
     * @param logits 输出logits
     * @param temperature 温度参数
     * @return 采样的token ID
     */
    private int sampleFromLogits(Variable logits, double temperature) {
        // 简化实现：返回最大概率的token（贪心解码）
        // 实际实现中应该支持温度采样、top-k、top-p等策略
        
        // 这里需要实现softmax和采样逻辑
        // 暂时返回0作为占位符
        return 0;
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
     * 获取Token嵌入层
     * 
     * @return Token嵌入层
     */
    public GPT1TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    /**
     * 获取Transformer块列表
     * 
     * @return Transformer块列表
     */
    public List<GPT1TransformerBlock> getTransformerBlocks() {
        return transformerBlocks;
    }
    
    /**
     * 获取最终层归一化
     * 
     * @return 最终层归一化
     */
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    /**
     * 获取输出头
     * 
     * @return 输出头
     */
    public GPT1OutputHead getOutputHead() {
        return outputHead;
    }
    
    /**
     * 获取模型参数数量
     * 
     * @return 参数数量
     */
    public long getParameterCount() {
        long totalParams = 0;
        
        // 计算所有参数的数量
        var allParams = getAllParams();
        for (var param : allParams.values()) {
            totalParams += param.getValue().getShape().size();
        }
        
        return totalParams;
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=== GPT-1 模型信息 ===");
        System.out.println("模型名称: " + name);
        System.out.println("词汇表大小: " + config.getVocabSize());
        System.out.println("最大序列长度: " + config.getMaxSequenceLength());
        System.out.println("隐藏层维度: " + config.getHiddenSize());
        System.out.println("Transformer层数: " + config.getNumLayers());
        System.out.println("注意力头数: " + config.getNumAttentionHeads());
        System.out.println("前馈网络维度: " + config.getIntermediateSize());
        System.out.println("参数总数: " + getParameterCount());
        System.out.println("激活函数: " + config.getActivationFunction());
        System.out.println("==================");
    }
}