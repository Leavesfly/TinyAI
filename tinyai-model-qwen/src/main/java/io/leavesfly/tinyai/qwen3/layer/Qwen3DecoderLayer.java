package io.leavesfly.tinyai.qwen3.layer;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.qwen3.Qwen3Config;

/**
 * Qwen3解码器层
 * 
 * 包含自注意力机制、前馈网络和残差连接：
 * 1. 输入 -> RMSNorm -> 自注意力 -> 残差连接
 * 2. 结果 -> RMSNorm -> MLP -> 残差连接 -> 输出
 * 
 * 这是Transformer解码器的标准结构，使用Pre-LN（预归一化）设计：
 * - 归一化在子层之前应用
 * - 有助于训练稳定性
 * - 现代大语言模型的标准选择
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3DecoderLayer extends Layer {
    
    /** 配置对象 */
    private Qwen3Config config;
    
    /** 隐藏维度 */
    private int hiddenSize;
    
    /** 自注意力层 */
    private Qwen3AttentionLayer selfAttention;
    
    /** MLP前馈网络 */
    private Qwen3MLPLayer mlp;
    
    /** 输入层归一化（注意力前） */
    private RMSNormLayer inputLayerNorm;
    
    /** 注意力后层归一化（MLP前） */
    private RMSNormLayer postAttentionLayerNorm;
    
    /**
     * 构造Qwen3解码器层
     * 
     * @param name 层名称
     * @param config Qwen3配置
     */
    public Qwen3DecoderLayer(String name, Qwen3Config config) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), 
              Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.hiddenSize = config.getHiddenSize();
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化自注意力层
            selfAttention = new Qwen3AttentionLayer(name + "_attention", config);
            
            // 初始化MLP层
            mlp = new Qwen3MLPLayer(name + "_mlp", config);
            
            // 初始化RMSNorm层
            inputLayerNorm = new RMSNormLayer(
                name + "_input_layernorm", hiddenSize, config.getRmsNormEps());
            postAttentionLayerNorm = new RMSNormLayer(
                name + "_post_attention_layernorm", hiddenSize, config.getRmsNormEps());
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];
        
        // 可选的注意力掩码
        Variable attentionMask = null;
        if (inputs.length > 1 && inputs[1] != null) {
            attentionMask = inputs[1];
        }
        
        return forwardDecoder(hiddenStates, attentionMask);
    }
    
    /**
     * 解码器层前向传播
     * 
     * @param hiddenStates 输入隐藏状态 [batch_size, seq_len, hidden_size]
     * @param attentionMask 注意力掩码
     * @return 解码器层输出
     */
    private Variable forwardDecoder(Variable hiddenStates, Variable attentionMask) {
        // 保存残差连接的输入
        Variable residual = hiddenStates;
        
        // 1. 预归一化 + 自注意力
        Variable normalizedInput = inputLayerNorm.layerForward(hiddenStates);
        
        Variable attentionOutput;
        if (attentionMask != null) {
            attentionOutput = selfAttention.layerForward(normalizedInput, attentionMask);
        } else {
            attentionOutput = selfAttention.layerForward(normalizedInput);
        }
        
        // 2. 残差连接
        Variable hiddenStatesAfterAttention = addResidualConnection(residual, attentionOutput);
        
        // 3. 预归一化 + MLP
        residual = hiddenStatesAfterAttention;
        Variable normalizedForMLP = postAttentionLayerNorm.layerForward(hiddenStatesAfterAttention);
        Variable mlpOutput = mlp.layerForward(normalizedForMLP);
        
        // 4. 残差连接
        Variable finalOutput = addResidualConnection(residual, mlpOutput);
        
        return finalOutput;
    }
    
    /**
     * 执行残差连接：output = input + residual
     * 
     * @param input 原始输入
     * @param residual 子层输出
     * @return 残差连接结果
     */
    private Variable addResidualConnection(Variable input, Variable residual) {
        NdArray inputData = input.getValue();
        NdArray residualData = residual.getValue();
        
        // 验证形状一致
        if (!inputData.getShape().equals(residualData.getShape())) {
            throw new IllegalArgumentException(
                String.format("残差连接要求输入形状 %s 与残差形状 %s 一致", 
                    inputData.getShape(), residualData.getShape()));
        }
        
        Shape shape = inputData.getShape();
        NdArray result = NdArray.of(shape);
        
        // 执行元素级加法
        if (shape.getDimNum() == 3) {
            int batchSize = shape.getDimension(0);
            int seqLen = shape.getDimension(1);
            int hiddenSize = shape.getDimension(2);
            
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int h = 0; h < hiddenSize; h++) {
                        float sum = inputData.get(b, s, h) + residualData.get(b, s, h);
                        result.set(sum, b, s, h);
                    }
                }
            }
        } else if (shape.getDimNum() == 2) {
            int rows = shape.getDimension(0);
            int cols = shape.getDimension(1);
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float sum = inputData.get(i, j) + residualData.get(i, j);
                    result.set(sum, i, j);
                }
            }
        } else {
            throw new IllegalArgumentException(
                String.format("残差连接不支持%dD张量", shape.getDimNum()));
        }
        
        return new Variable(result);
    }
    
    /**
     * 获取隐藏维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * 获取自注意力层
     */
    public Qwen3AttentionLayer getSelfAttention() {
        return selfAttention;
    }
    
    /**
     * 获取MLP层
     */
    public Qwen3MLPLayer getMlp() {
        return mlp;
    }
    
    /**
     * 获取输入层归一化
     */
    public RMSNormLayer getInputLayerNorm() {
        return inputLayerNorm;
    }
    
    /**
     * 获取注意力后层归一化
     */
    public RMSNormLayer getPostAttentionLayerNorm() {
        return postAttentionLayerNorm;
    }
}