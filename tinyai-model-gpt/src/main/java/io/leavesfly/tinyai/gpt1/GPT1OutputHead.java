package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-1 输出头实现
 * 
 * 负责将Transformer的隐藏表示转换为词汇表上的概率分布
 * 通过线性投影层将hiddenSize维度映射到vocabSize维度
 * 
 * 在GPT-1中，输出层通常与Token嵌入层共享权重（权重绑定）
 * 这样可以减少参数量并提高训练效果
 * 
 * @author 山泽
 * @version 1.0
 */
public class GPT1OutputHead extends Layer {
    
    /** 输出投影权重矩阵 (hiddenSize, vocabSize) */
    private Parameter outputProjection;
    
    /** 输出偏置向量 (vocabSize) - 可选 */
    private Parameter outputBias;
    
    /** 配置信息 */
    private GPT1Config config;
    
    /** 是否使用偏置 */
    private boolean useBias;
    
    /** 是否与Token嵌入共享权重 */
    private boolean shareEmbeddingWeights;
    
    /** 共享的Token嵌入参数（如果启用权重共享） */
    private Parameter sharedTokenEmbedding;
    
    /**
     * 构造GPT-1输出头
     * 
     * @param name 层名称
     * @param config GPT-1配置
     * @param useBias 是否使用偏置
     */
    public GPT1OutputHead(String name, GPT1Config config, boolean useBias) {
        super(name, 
              Shape.of(-1, -1, config.getHiddenSize()), 
              Shape.of(-1, -1, config.getVocabSize()));
        
        this.config = config;
        this.useBias = useBias;
        this.shareEmbeddingWeights = false;
        
        init();
    }
    
    /**
     * 构造GPT-1输出头（默认不使用偏置）
     * 
     * @param name 层名称
     * @param config GPT-1配置
     */
    public GPT1OutputHead(String name, GPT1Config config) {
        this(name, config, false);
    }
    
    /**
     * 构造支持权重共享的GPT-1输出头
     * 
     * @param name 层名称
     * @param config GPT-1配置
     * @param useBias 是否使用偏置
     * @param tokenEmbedding 共享的Token嵌入权重
     */
    public GPT1OutputHead(String name, GPT1Config config, boolean useBias, Parameter tokenEmbedding) {
        this(name, config, useBias);
        this.shareEmbeddingWeights = true;
        this.sharedTokenEmbedding = tokenEmbedding;
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            if (shareEmbeddingWeights && sharedTokenEmbedding != null) {
                // 使用共享的Token嵌入权重
                this.outputProjection = sharedTokenEmbedding;
            } else {
                // 独立初始化输出投影权重
                this.outputProjection = new Parameter(
                    NdArray.likeRandomN(Shape.of(config.getHiddenSize(), config.getVocabSize()))
                           .mulNum((float) config.getInitializerRange())
                );
                outputProjection.setName(name + "_output_projection");
                addParam(outputProjection.getName(), outputProjection);
            }
            
            // 如果使用偏置，初始化偏置参数
            if (useBias) {
                this.outputBias = new Parameter(
                    NdArray.zeros(Shape.of(config.getVocabSize()))
                );
                outputBias.setName(name + "_output_bias");
                addParam(outputBias.getName(), outputBias);
            }
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];  // shape: (batchSize, sequenceLength, hiddenSize)
        NdArray hiddenData = hiddenStates.getValue();
        
        int batchSize = hiddenData.getShape().getDimension(0);
        int sequenceLength = hiddenData.getShape().getDimension(1);
        int hiddenSize = hiddenData.getShape().getDimension(2);
        
        // 验证输入维度
        if (hiddenSize != config.getHiddenSize()) {
            throw new IllegalArgumentException(
                String.format("输入隐藏维度 %d 与配置的隐藏维度 %d 不匹配", 
                            hiddenSize, config.getHiddenSize())
            );
        }
        
        // 执行线性变换：hidden_states @ output_projection + bias
        Variable logits = computeLinearProjection(hiddenStates, batchSize, sequenceLength);
        
        return logits;
    }
    
    /**
     * 计算线性投影
     * 
     * @param hiddenStates 隐藏状态
     * @param batchSize 批次大小
     * @param sequenceLength 序列长度
     * @return 投影后的logits
     */
    private Variable computeLinearProjection(Variable hiddenStates, int batchSize, int sequenceLength) {
        NdArray hiddenData = hiddenStates.getValue();
        NdArray logits = NdArray.of(Shape.of(batchSize, sequenceLength, config.getVocabSize()));
        
        // 执行矩阵乘法：(batch_size, seq_len, hidden_size) @ (hidden_size, vocab_size)
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sequenceLength; s++) {
                for (int v = 0; v < config.getVocabSize(); v++) {
                    float sum = 0.0f;
                    
                    // 点积计算
                    for (int h = 0; h < config.getHiddenSize(); h++) {
                        float hiddenValue = hiddenData.get(b, s, h);
                        float weightValue;
                        
                        if (shareEmbeddingWeights) {
                            // 权重共享时，需要转置Token嵌入矩阵
                            weightValue = outputProjection.getValue().get(v, h);
                        } else {
                            weightValue = outputProjection.getValue().get(h, v);
                        }
                        
                        sum += hiddenValue * weightValue;
                    }
                    
                    // 添加偏置（如果使用）
                    if (useBias) {
                        sum += outputBias.getValue().get(v);
                    }
                    
                    logits.set(sum, b, s, v);
                }
            }
        }
        
        return new Variable(logits);
    }
    
    /**
     * 设置权重共享
     * 
     * @param tokenEmbedding 要共享的Token嵌入权重
     */
    public void setSharedEmbeddingWeights(Parameter tokenEmbedding) {
        if (tokenEmbedding.getValue().getShape().getDimension(0) != config.getVocabSize() ||
            tokenEmbedding.getValue().getShape().getDimension(1) != config.getHiddenSize()) {
            throw new IllegalArgumentException("Token嵌入权重的形状与配置不匹配");
        }
        
        this.shareEmbeddingWeights = true;
        this.sharedTokenEmbedding = tokenEmbedding;
        this.outputProjection = tokenEmbedding;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 简化的反向传播实现
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // ==================== Getter方法 ====================
    
    /**
     * 获取输出投影权重
     * 
     * @return 输出投影权重
     */
    public Parameter getOutputProjection() {
        return outputProjection;
    }
    
    /**
     * 获取输出偏置
     * 
     * @return 输出偏置
     */
    public Parameter getOutputBias() {
        return outputBias;
    }
    
    /**
     * 获取配置信息
     * 
     * @return GPT-1配置
     */
    public GPT1Config getConfig() {
        return config;
    }
    
    /**
     * 是否使用偏置
     * 
     * @return 是否使用偏置
     */
    public boolean isUseBias() {
        return useBias;
    }
    
    /**
     * 是否共享嵌入权重
     * 
     * @return 是否共享嵌入权重
     */
    public boolean isShareEmbeddingWeights() {
        return shareEmbeddingWeights;
    }
    
    /**
     * 获取共享的Token嵌入参数
     * 
     * @return 共享的Token嵌入参数
     */
    public Parameter getSharedTokenEmbedding() {
        return sharedTokenEmbedding;
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
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return config.getHiddenSize();
    }
}