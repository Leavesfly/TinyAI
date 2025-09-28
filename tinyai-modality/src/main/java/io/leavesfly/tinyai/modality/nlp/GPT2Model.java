package io.leavesfly.tinyai.modality.nlp;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.block.transformer.GPT2Block;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2OutputHead;
import io.leavesfly.tinyai.nnet.layer.transformer.GPT2TokenEmbedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-2 小规模语言模型实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * GPT2Model类实现了GPT-2语言模型，基于Transformer解码器的自回归语言模型。
 * 特点：
 * 1. 仅使用解码器架构
 * 2. 使用掩码多头自注意力防止未来信息泄露
 * 3. Pre-LayerNorm结构
 * 4. 残差连接
 * <p>
 * 模型结构：
 * Token Embedding + Position Embedding
 * → N × GPT2Block
 * → Final LayerNorm
 * → Output Head
 * <p>
 */
public class GPT2Model extends Block {
    
    // 模型超参数
    private int vocabSize;      // 词汇表大小
    private int dModel;         // 模型维度
    private int numLayers;      // Transformer块数量
    private int numHeads;       // 注意力头数量
    private int dFF;            // 前馈网络隐藏维度
    private int maxSeqLength;   // 最大序列长度
    private double dropoutRate; // Dropout比率
    
    // 模型组件
    private GPT2TokenEmbedding tokenEmbedding;  // Token嵌入层
    private List<GPT2Block> transformerBlocks;  // Transformer块列表
    private LayerNorm finalLayerNorm;           // 最终层归一化
    private GPT2OutputHead outputHead;          // 输出头

    /**
     * 构造GPT-2模型
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer块数量
     * @param numHeads 注意力头数量
     * @param dFF 前馈网络隐藏维度
     * @param maxSeqLength 最大序列长度
     * @param dropoutRate Dropout比率
     */
    public GPT2Model(String name, int vocabSize, int dModel, int numLayers, 
                     int numHeads, int dFF, int maxSeqLength, double dropoutRate) {
        super(name, Shape.of(-1, maxSeqLength), Shape.of(-1, maxSeqLength, vocabSize));
        
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.maxSeqLength = maxSeqLength;
        this.dropoutRate = dropoutRate;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @param dModel 模型维度
     * @param numLayers Transformer块数量
     * @param maxSeqLength 最大序列长度
     */
    public GPT2Model(String name, int vocabSize, int dModel, int numLayers, int maxSeqLength) {
        this(name, vocabSize, dModel, numLayers, 8, dModel * 4, maxSeqLength, 0.1);
    }
    
    /**
     * 小型GPT-2配置的构造函数
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小 
     * @param maxSeqLength 最大序列长度
     */
    public GPT2Model(String name, int vocabSize, int maxSeqLength) {
        this(name, vocabSize, 768, 12, 12, 3072, maxSeqLength, 0.1);
    }
    
    /**
     * 兼容原有构造函数
     */
    public GPT2Model(String _name, Shape _inputShape) {
        super(_name, _inputShape);
        // 使用默认配置
        this.vocabSize = 50257;  // GPT-2默认词汇表大小
        this.dModel = 768;
        this.numLayers = 12;
        this.numHeads = 12;
        this.dFF = 3072;
        this.maxSeqLength = 1024;
        this.dropoutRate = 0.1;
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化Token嵌入层
            tokenEmbedding = new GPT2TokenEmbedding(
                name + "_token_embedding", 
                vocabSize, 
                dModel, 
                maxSeqLength, 
                true,  // 使用位置嵌入
                dropoutRate
            );
            addLayer(tokenEmbedding);
            
            // 2. 初始化Transformer块列表
            transformerBlocks = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                GPT2Block block = new GPT2Block(
                    name + "_block_" + i,
                    dModel,
                    numHeads,
                    dFF,
                    dropoutRate
                );
                transformerBlocks.add(block);
                addLayer(block);
            }
            
            // 3. 初始化最终层归一化
            finalLayerNorm = new LayerNorm(name + "_final_ln", dModel);
            addLayer(finalLayerNorm);
            
            // 4. 初始化输出头
            outputHead = new GPT2OutputHead(
                name + "_output_head",
                dModel,
                vocabSize,
                false  // GPT-2通常不使用输出偏置
            );
            addLayer(outputHead);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // shape: (batch_size, seq_len)
        
        // 验证输入序列长度
        int seqLen = input.getValue().getShape().getDimension(1);
        if (seqLen > maxSeqLength) {
            throw new IllegalArgumentException(
                String.format("输入序列长度 %d 超过最大长度 %d", seqLen, maxSeqLength)
            );
        }
        
        // 1. Token嵌入 + 位置嵌入
        Variable x = tokenEmbedding.layerForward(input);  // shape: (batch_size, seq_len, dModel)
        
        // 2. 通过所有Transformer块
        for (GPT2Block block : transformerBlocks) {
            x = block.layerForward(x);
        }
        
        // 3. 最终层归一化
        x = finalLayerNorm.layerForward(x);
        
        // 4. 输出头得到词汇表概率分布
        Variable output = outputHead.layerForward(x);  // shape: (batch_size, seq_len, vocab_size)
        
        return output;
    }
    
    /**
     * 获取Token嵌入层
     */
    public GPT2TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }
    
    /**
     * 获取Transformer块列表
     */
    public List<GPT2Block> getTransformerBlocks() {
        return transformerBlocks;
    }
    
    /**
     * 获取指定索引的Transformer块
     */
    public GPT2Block getTransformerBlock(int index) {
        if (index < 0 || index >= transformerBlocks.size()) {
            throw new IndexOutOfBoundsException("Transformer块索引超出范围: " + index);
        }
        return transformerBlocks.get(index);
    }
    
    /**
     * 获取最终层归一化
     */
    public LayerNorm getFinalLayerNorm() {
        return finalLayerNorm;
    }
    
    /**
     * 获取输出头
     */
    public GPT2OutputHead getOutputHead() {
        return outputHead;
    }
    
    /**
     * 获取模型配置信息
     */
    public String getModelConfig() {
        return String.format(
            "GPT2Model Config:\n" +
            "  - Vocab Size: %d\n" +
            "  - Model Dim: %d\n" +
            "  - Num Layers: %d\n" +
            "  - Num Heads: %d\n" +
            "  - FFN Dim: %d\n" +
            "  - Max Seq Length: %d\n" +
            "  - Dropout Rate: %.2f",
            vocabSize, dModel, numLayers, numHeads, dFF, maxSeqLength, dropoutRate
        );
    }
    
    /**
     * 创建小型GPT-2模型的工厂方法
     * 
     * @param name 模型名称
     * @param vocabSize 词汇表大小
     * @return 小型GPT-2模型实例
     */
    public static GPT2Model createTinyModel(String name, int vocabSize) {
        // 使用小型配置参数
        return new GPT2Model(
            name,
            vocabSize,
            384,      // dModel: 小型模型维度
            6,        // numLayers: 6层
            6,        // numHeads: 6个注意力头
            1536,     // dFF: 前馈网络维度
            512,      // maxSeqLength: 最大序列长度
            0.1       // dropoutRate: dropout比率
        );
    }
    
    /**
     * 打印模型配置信息
     */
    public void printModelInfo() {
        System.out.println("\n=== GPT-2 Model Information ===");
        System.out.println(getModelConfig());
        System.out.println("Model Parameters:");
        
        // 统计参数数量
        long totalParams = 0;
        
        // Token嵌入层参数
        long tokenEmbedParams = (long) vocabSize * dModel;  // 词嵌入
        long posEmbedParams = (long) maxSeqLength * dModel;  // 位置嵌入
        totalParams += tokenEmbedParams + posEmbedParams;
        
        // Transformer块参数
        for (int i = 0; i < numLayers; i++) {
            // 自注意力参数
            long attnParams = (long) dModel * dModel * 4;  // Q,K,V,O矩阵
            // 前馈网络参数
            long ffnParams = (long) dModel * dFF * 2;      // 两个线性层
            // 层归一化参数
            long lnParams = (long) dModel * 2 * 2;         // 两个LayerNorm层
            totalParams += attnParams + ffnParams + lnParams;
        }
        
        // 最终层归一化参数
        totalParams += (long) dModel * 2;
        
        // 输出头参数（通常与词嵌入共享权重，这里单独计算）
        long outputHeadParams = (long) dModel * vocabSize;
        totalParams += outputHeadParams;
        
        System.out.println("  - Token Embedding: " + formatNumber(tokenEmbedParams));
        System.out.println("  - Position Embedding: " + formatNumber(posEmbedParams));
        System.out.println("  - Transformer Blocks (" + numLayers + "): " + formatNumber(totalParams - tokenEmbedParams - posEmbedParams - dModel * 2 - outputHeadParams));
        System.out.println("  - Final LayerNorm: " + formatNumber(dModel * 2));
        System.out.println("  - Output Head: " + formatNumber(outputHeadParams));
        System.out.println("  - Total Parameters: " + formatNumber(totalParams));
        System.out.println("==============================\n");
    }
    
    /**
     * 格式化数字显示（添加千分位分隔符）
     */
    private String formatNumber(long number) {
        return String.format("%,d", number);
    }
    
    /**
     * 预测下一个token
     * 
     * @param input 输入序列，形状为 (1, seq_len)
     * @return 预测的下一个token ID
     */
    public int predictNextToken(NdArray input) {
        try {
            // 前向传播
            Variable inputVar = new Variable(input);
            Variable output = layerForward(inputVar);  // shape: (1, seq_len, vocab_size)
            
            // 获取最后一个时间步的输出
            NdArray logits = output.getValue();
            int seqLen = logits.getShape().getDimension(1);
            
            // 提取最后一个位置的logits: (vocab_size,)
            NdArray lastLogits = NdArray.of(Shape.of(vocabSize));
            for (int i = 0; i < vocabSize; i++) {
                lastLogits.set(logits.get(0, seqLen - 1, i), i);
            }
            
            // 使用softmax获取概率分布
            NdArray probabilities = applySoftmax(lastLogits);
            
            // 使用贪婪搜索选择最高概率的token
            int maxIndex = 0;
            float maxProb = probabilities.get(0);
            
            for (int i = 1; i < vocabSize; i++) {
                float prob = probabilities.get(i);
                if (prob > maxProb) {
                    maxProb = prob;
                    maxIndex = i;
                }
            }
            
            return maxIndex;
            
        } catch (Exception e) {
            System.err.println("Error in predictNextToken: " + e.getMessage());
            // 返回未知token ID作为fallback
            return SimpleTokenizer.UNK_ID;
        }
    }
    
    /**
     * 应用softmax函数
     * 
     * @param logits 输入logits
     * @return softmax概率分布
     */
    private NdArray applySoftmax(NdArray logits) {
        NdArray result = NdArray.of(logits.getShape());
        
        // 找到最大值用于数值稳定性
        float maxLogit = logits.get(0);
        for (int i = 1; i < logits.getShape().size(); i++) {
            maxLogit = Math.max(maxLogit, logits.get(i));
        }
        
        // 计算exp(x - max)
        float sum = 0.0f;
        for (int i = 0; i < logits.getShape().size(); i++) {
            float expVal = (float) Math.exp(logits.get(i) - maxLogit);
            result.set(expVal, i);
            sum += expVal;
        }
        
        // 归一化
        for (int i = 0; i < logits.getShape().size(); i++) {
            result.set(result.get(i) / sum, i);
        }
        
        return result;
    }

    // Getter方法
    public int getVocabSize() { return vocabSize; }
    public int getDModel() { return dModel; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getDFF() { return dFF; }
    public int getMaxSeqLength() { return maxSeqLength; }
    public double getDropoutRate() { return dropoutRate; }
}