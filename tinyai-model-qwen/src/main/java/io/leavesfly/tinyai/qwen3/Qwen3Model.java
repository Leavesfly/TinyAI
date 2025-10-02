package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.Model;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

/**
 * Qwen3模型
 * 
 * 完整的Qwen3大语言模型实现，继承自TinyAI的Model类。
 * 该类封装了Qwen3Block并添加了语言模型头（LM Head），
 * 提供完整的因果语言建模功能。
 * 
 * 模型结构：
 * Input(token_ids) -> Qwen3Block -> LMHead -> Logits
 * 
 * 主要功能：
 * 1. 前向传播：计算下一个token的概率分布
 * 2. 文本生成：支持自回归生成
 * 3. 模型管理：保存、加载、信息显示
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Model extends Model {
    
    /**
     * 配置信息
     */
    private Qwen3Config config;
    
    /**
     * 核心Qwen3块
     */
    private Qwen3Block qwen3Block;
    
    /**
     * 语言模型头（输出投影层）
     */
    private LinearLayer lmHead;

    /**
     * 构造Qwen3模型
     * 
     * @param name 模型名称
     * @param config 配置信息
     */
    public Qwen3Model(String name, Qwen3Config config) {
        super(name, createQwen3Block(name, config));
        
        this.config = config;
        this.qwen3Block = (Qwen3Block) getBlock();
        
        // 初始化语言模型头
        initLMHead();
        
        // 设置模型描述
        setDescription("Qwen3大语言模型 - " + config.getNumHiddenLayers() + "层Transformer");
    }
    
    /**
     * 创建Qwen3Block
     * 
     * @param name 名称
     * @param config 配置
     * @return Qwen3Block实例
     */
    private static Qwen3Block createQwen3Block(String name, Qwen3Config config) {
        return new Qwen3Block(name + "_qwen3", config);
    }
    
    /**
     * 初始化语言模型头
     */
    private void initLMHead() {
        // LM Head：将隐藏状态映射到词汇表概率分布
        lmHead = new LinearLayer(
            getName() + "_lm_head", 
            config.getHiddenSize(), 
            config.getVocabSize(), 
            false // 不使用偏置
        );
    }

    /**
     * 模型前向传播
     * 
     * @param inputIds 输入token序列，形状为(batch_size, seq_len)或(seq_len,)
     * @return 输出logits，形状为(batch_size, seq_len, vocab_size)
     */
    public Variable forwardWithLogits(Variable inputIds) {
        // 1. 通过Qwen3Block获得隐藏状态
        Variable hiddenStates = qwen3Block.layerForward(inputIds);
        
        // 2. 通过语言模型头获得logits
        Variable logits = computeLogits(hiddenStates);
        
        return logits;
    }
    
    /**
     * 计算logits
     * 
     * @param hiddenStates 隐藏状态
     * @return logits
     */
    private Variable computeLogits(Variable hiddenStates) {
        NdArray hiddenData = hiddenStates.getValue();
        Shape hiddenShape = hiddenData.getShape();
        
        // 处理不同维度的输入
        if (hiddenShape.getDimNum() == 2) {
            // 2D输入: (batch_size, hidden_size) 
            return lmHead.layerForward(hiddenStates);
        } else if (hiddenShape.getDimNum() == 3) {
            // 3D输入: (batch_size, seq_len, hidden_size)
            int batchSize = hiddenShape.getDimension(0);
            int seqLen = hiddenShape.getDimension(1);
            int hiddenSize = hiddenShape.getDimension(2);
            
            // 重塑为2D进行线性变换
            NdArray reshaped2D = hiddenData.reshape(Shape.of(batchSize * seqLen, hiddenSize));
            Variable logits2D = lmHead.layerForward(new Variable(reshaped2D));
            
            // 重塑回3D
            NdArray logits3D = logits2D.getValue().reshape(Shape.of(batchSize, seqLen, config.getVocabSize()));
            return new Variable(logits3D);
        } else {
            throw new IllegalArgumentException("不支持的隐藏状态维度: " + hiddenShape.getDimNum());
        }
    }
    
    /**
     * 预测下一个token
     * 
     * @param inputIds 输入token序列
     * @return 下一个token的ID
     */
    public int predictNextToken(NdArray inputIds) {
        Variable logits = forwardWithLogits(new Variable(inputIds));
        NdArray logitsData = logits.getValue();
        
        // 获取最后一个位置的logits
        NdArray lastLogits = getLastTokenLogits(logitsData);
        
        // 找到概率最大的token
        return argmax(lastLogits);
    }
    
    /**
     * 获取最后一个token位置的logits
     * 
     * @param logits 完整的logits张量
     * @return 最后位置的logits
     */
    private NdArray getLastTokenLogits(NdArray logits) {
        Shape logitsShape = logits.getShape();
        
        if (logitsShape.getDimNum() == 2) {
            // (batch_size, vocab_size) - 已经是最后位置
            return logits;
        } else if (logitsShape.getDimNum() == 3) {
            // (batch_size, seq_len, vocab_size) - 取最后一个位置
            int batchSize = logitsShape.getDimension(0);
            int seqLen = logitsShape.getDimension(1);
            int vocabSize = logitsShape.getDimension(2);
            
            NdArray lastLogits = NdArray.of(Shape.of(batchSize, vocabSize));
            
            for (int b = 0; b < batchSize; b++) {
                for (int v = 0; v < vocabSize; v++) {
                    float value = logits.get(b, seqLen - 1, v);
                    lastLogits.set(value, b, v);
                }
            }
            
            return lastLogits;
        } else {
            throw new IllegalArgumentException("不支持的logits维度: " + logitsShape.getDimNum());
        }
    }
    
    /**
     * 找到最大值的索引（argmax）
     * 
     * @param array 输入数组
     * @return 最大值的索引
     */
    private int argmax(NdArray array) {
        NdArray flattenedArray = array.flatten();
        
        // 获取底层数据
        float[] data;
        if (flattenedArray instanceof io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) {
            data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) flattenedArray).buffer;
        } else {
            // 对于非 NdArrayCpu 实现，使用 get 方法
            Shape shape = flattenedArray.getShape();
            data = new float[shape.size()];
            for (int i = 0; i < shape.size(); i++) {
                data[i] = flattenedArray.get(i);
            }
        }
        
        int maxIndex = 0;
        float maxValue = data[0];
        
        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxValue) {
                maxValue = data[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * 简单的贪心文本生成
     * 
     * @param inputIds 初始输入序列
     * @param maxLength 生成的最大长度
     * @return 完整的生成序列
     */
    public NdArray generate(NdArray inputIds, int maxLength) {
        // 创建输入副本，因为 NdArray 接口没有 copy() 方法
        NdArray currentSequence = createCopy(inputIds);
        Shape currentShape = currentSequence.getShape();
        
        // 确保输入是2D格式 (batch_size, seq_len)
        if (currentShape.getDimNum() == 1) {
            currentSequence = currentSequence.reshape(Shape.of(1, currentShape.getDimension(0)));
        }
        
        int batchSize = currentSequence.getShape().getDimension(0);
        int currentLen = currentSequence.getShape().getDimension(1);
        
        for (int step = currentLen; step < maxLength; step++) {
            // 预测下一个token
            Variable logits = forwardWithLogits(new Variable(currentSequence));
            NdArray lastLogits = getLastTokenLogits(logits.getValue());
            
            // 对每个batch选择最佳token
            int[] nextTokens = new int[batchSize];
            for (int b = 0; b < batchSize; b++) {
                NdArray batchLogits = NdArray.of(Shape.of(config.getVocabSize()));
                for (int v = 0; v < config.getVocabSize(); v++) {
                    batchLogits.set(lastLogits.get(b, v), v);
                }
                nextTokens[b] = argmax(batchLogits);
            }
            
            // 扩展序列
            currentSequence = appendTokens(currentSequence, nextTokens);
            
            // 检查结束条件（可以添加EOS token检查）
            boolean shouldStop = false;
            for (int token : nextTokens) {
                if (token == config.getEosTokenId()) {
                    shouldStop = true;
                    break;
                }
            }
            if (shouldStop) break;
        }
        
        return currentSequence;
    }
    
    /**
     * 在序列末尾添加新tokens
     * 
     * @param sequence 当前序列
     * @param newTokens 新tokens
     * @return 扩展后的序列
     */
    private NdArray appendTokens(NdArray sequence, int[] newTokens) {
        int batchSize = sequence.getShape().getDimension(0);
        int seqLen = sequence.getShape().getDimension(1);
        
        NdArray newSequence = NdArray.of(Shape.of(batchSize, seqLen + 1));
        
        // 复制原序列
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                newSequence.set(sequence.get(b, s), b, s);
            }
            // 添加新token
            newSequence.set(newTokens[b], b, seqLen);
        }
        
        return newSequence;
    }
    
    /**
     * 创建 NdArray 的副本
     * 
     * @param original 原始数组
     * @return 副本数组
     */
    private NdArray createCopy(NdArray original) {
        Shape shape = original.getShape();
        NdArray copy = NdArray.of(shape);
        
        // 复制数据
        for (int i = 0; i < shape.size(); i++) {
            int[] indices = convertLinearToMultiIndex(i, shape);
            copy.set(original.get(indices), indices);
        }
        
        return copy;
    }
    
    /**
     * 将线性索引转换为多维索引
     * 
     * @param linearIndex 线性索引
     * @param shape 数组形状
     * @return 多维索引数组
     */
    private int[] convertLinearToMultiIndex(int linearIndex, Shape shape) {
        int[] indices = new int[shape.getDimNum()];
        int remaining = linearIndex;
        
        for (int i = shape.getDimNum() - 1; i >= 0; i--) {
            indices[i] = remaining % shape.getDimension(i);
            remaining /= shape.getDimension(i);
        }
        
        return indices;
    }
    
    /**
     * 创建用于演示的小型模型
     * 
     * @param name 模型名称
     * @return 小型Qwen3模型实例
     */
    public static Qwen3Model createTinyModel(String name) {
        Qwen3Config tinyConfig = Qwen3Config.createTinyConfig();
        return new Qwen3Model(name, tinyConfig);
    }
    
    /**
     * 打印模型信息
     */
    public void printModelInfo() {
        System.out.println("=== Qwen3模型信息 ===");
        System.out.println("模型名称: " + getName());
        System.out.println("配置信息: " + config.toString());
        System.out.println();
        
        // 打印架构信息
        qwen3Block.printArchitecture();
        
        // 打印模型统计
        System.out.println(qwen3Block.getModelStats());
    }
    
    // Getters
    
    /**
     * 获取配置信息
     * 
     * @return 配置信息
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取Qwen3核心块
     * 
     * @return Qwen3核心块
     */
    public Qwen3Block getQwen3Block() {
        return qwen3Block;
    }
    
    /**
     * 获取语言模型头
     * 
     * @return 语言模型头
     */
    public LinearLayer getLmHead() {
        return lmHead;
    }
}