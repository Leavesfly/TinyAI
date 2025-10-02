package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3多头注意力机制层
 * 
 * 实现了Qwen3模型的多头注意力机制，支持以下特性：
 * 1. 分组查询注意力（Grouped Query Attention, GQA）
 * 2. 旋转位置编码（RoPE）
 * 3. 因果掩码（用于自回归生成）
 * 4. 缩放点积注意力
 * 
 * 相比标准的多头注意力，Qwen3使用了GQA来减少KV缓存的内存占用，
 * 同时保持模型性能。
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3Attention extends Layer {
    
    /**
     * 配置信息
     */
    private Qwen3Config config;
    
    /**
     * 层索引
     */
    private int layerIdx;
    
    /**
     * 隐藏层维度
     */
    private int hiddenSize;
    
    /**
     * 注意力头数
     */
    private int numHeads;
    
    /**
     * 每个注意力头的维度
     */
    private int headDim;
    
    /**
     * KV头数（用于分组查询注意力）
     */
    private int numKeyValueHeads;
    
    /**
     * KV头分组数
     */
    private int numKeyValueGroups;
    
    /**
     * 最大位置编码长度
     */
    private int maxPositionEmbeddings;
    
    /**
     * Q投影层
     */
    private LinearLayer qProj;
    
    /**
     * K投影层
     */
    private LinearLayer kProj;
    
    /**
     * V投影层
     */
    private LinearLayer vProj;
    
    /**
     * 输出投影层
     */
    private LinearLayer oProj;
    
    /**
     * 旋转位置编码
     */
    private RotaryPositionalEmbedding rotaryEmb;
    
    /**
     * 是否使用因果掩码
     */
    private boolean useMask;

    /**
     * 构造Qwen3注意力层
     * 
     * @param name 层名称
     * @param config 配置信息
     * @param layerIdx 层索引
     * @param useMask 是否使用因果掩码
     */
    public Qwen3Attention(String name, Qwen3Config config, int layerIdx, boolean useMask) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.layerIdx = layerIdx;
        this.useMask = useMask;
        
        this.hiddenSize = config.getHiddenSize();
        this.numHeads = config.getNumAttentionHeads();
        this.headDim = config.getHeadDim();
        this.numKeyValueHeads = config.getNumKeyValueHeads();
        this.numKeyValueGroups = config.getNumKeyValueGroups();
        this.maxPositionEmbeddings = config.getMaxPositionEmbeddings();
        
        if ((this.headDim * this.numHeads) != this.hiddenSize) {
            throw new IllegalArgumentException(
                "hiddenSize必须能被numHeads整除 (得到 " + this.hiddenSize + " 和 " + this.numHeads + ")");
        }
        
        init();
    }

    /**
     * 使用默认参数的构造函数
     */
    public Qwen3Attention(String name, Qwen3Config config, int layerIdx) {
        this(name, config, layerIdx, true);
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化Q、K、V投影层
            qProj = new LinearLayer(name + "_q_proj", hiddenSize, numHeads * headDim, true);
            kProj = new LinearLayer(name + "_k_proj", hiddenSize, numKeyValueHeads * headDim, true);
            vProj = new LinearLayer(name + "_v_proj", hiddenSize, numKeyValueHeads * headDim, true);
            oProj = new LinearLayer(name + "_o_proj", numHeads * headDim, hiddenSize, false);
            
            // 初始化旋转位置编码
            rotaryEmb = new RotaryPositionalEmbedding(
                name + "_rotary_emb", 
                headDim, 
                maxPositionEmbeddings, 
                config.getRopeTheta()
            );
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];
        // 可选的注意力掩码和位置ID参数暂时忽略，使用默认行为
        
        NdArray hiddenData = hiddenStates.getValue();
        int batchSize = hiddenData.getShape().getDimension(0);
        int seqLen = hiddenData.getShape().getDimension(1);
        
        // 计算Q、K、V
        Variable queryStates = computeQKV(hiddenStates, qProj, batchSize, seqLen, numHeads, headDim);
        Variable keyStates = computeQKV(hiddenStates, kProj, batchSize, seqLen, numKeyValueHeads, headDim);
        Variable valueStates = computeQKV(hiddenStates, vProj, batchSize, seqLen, numKeyValueHeads, headDim);
        
        // 应用旋转位置编码
        queryStates = applyRoPE(queryStates, seqLen);
        keyStates = applyRoPE(keyStates, seqLen);
        
        // 重复KV以匹配Q的头数（分组查询注意力）
        keyStates = repeatKV(keyStates, numKeyValueGroups, batchSize, seqLen, numHeads, headDim);
        valueStates = repeatKV(valueStates, numKeyValueGroups, batchSize, seqLen, numHeads, headDim);
        
        // 计算注意力
        Variable attnOutput = computeAttention(queryStates, keyStates, valueStates, batchSize, seqLen);
        
        // 重塑为二维进行最终投影
        NdArray attnData = attnOutput.getValue();
        NdArray reshaped2D = attnData.reshape(Shape.of(batchSize * seqLen, hiddenSize));
        Variable finalOutput = oProj.layerForward(new Variable(reshaped2D));
        
        // 重塑回三维
        NdArray result = finalOutput.getValue().reshape(Shape.of(batchSize, seqLen, hiddenSize));
        return new Variable(result);
    }
    
    /**
     * 计算Q、K或V矩阵
     */
    private Variable computeQKV(Variable hiddenStates, LinearLayer projLayer, 
                                int batchSize, int seqLen, int numHeads, int headDim) {
        // 重塑为二维进行线性变换
        NdArray hiddenData = hiddenStates.getValue();
        NdArray reshaped2D = hiddenData.reshape(Shape.of(batchSize * seqLen, hiddenSize));
        
        // 线性变换
        Variable projected = projLayer.layerForward(new Variable(reshaped2D));
        
        // 重塑为多头形式: (batch_size, seq_len, num_heads, head_dim)
        // 然后转置为: (batch_size, num_heads, seq_len, head_dim)
        NdArray projectedData = projected.getValue();
        NdArray reshaped4D = projectedData.reshape(Shape.of(batchSize, seqLen, numHeads, headDim));
        NdArray transposed = transpose4D(reshaped4D, batchSize, numHeads, seqLen, headDim);
        
        return new Variable(transposed);
    }
    
    /**
     * 4D张量转置：(batch, seq, heads, dim) -> (batch, heads, seq, dim)
     */
    private NdArray transpose4D(NdArray input, int batchSize, int numHeads, int seqLen, int headDim) {
        NdArray output = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = input.get(b, s, h, d);
                        output.set(value, b, h, s, d);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * 应用旋转位置编码
     */
    private Variable applyRoPE(Variable states, int seqLen) {
        return rotaryEmb.layerForward(states, new Variable(NdArray.of(new float[]{seqLen})));
    }
    
    /**
     * 重复KV张量以匹配查询头数（分组查询注意力）
     */
    private Variable repeatKV(Variable hiddenStates, int nRep, int batchSize, int seqLen, int targetHeads, int headDim) {
        if (nRep == 1) {
            return hiddenStates;
        }
        
        NdArray input = hiddenStates.getValue();
        int originalHeads = input.getShape().getDimension(1);
        
        NdArray output = NdArray.of(Shape.of(batchSize, targetHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < originalHeads; h++) {
                for (int rep = 0; rep < nRep; rep++) {
                    int targetHead = h * nRep + rep;
                    for (int s = 0; s < seqLen; s++) {
                        for (int d = 0; d < headDim; d++) {
                            float value = input.get(b, h, s, d);
                            output.set(value, b, targetHead, s, d);
                        }
                    }
                }
            }
        }
        
        return new Variable(output);
    }
    
    /**
     * 计算缩放点积注意力
     */
    private Variable computeAttention(Variable query, Variable key, Variable value, int batchSize, int seqLen) {
        NdArray queryData = query.getValue();
        NdArray keyData = key.getValue();
        NdArray valueData = value.getValue();
        
        double scale = 1.0 / Math.sqrt(headDim);
        NdArray attnOutput = NdArray.of(Shape.of(batchSize, numHeads, seqLen, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // 计算attention scores: Q * K^T
                NdArray scores = NdArray.of(Shape.of(seqLen, seqLen));
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            score += queryData.get(b, h, i, d) * keyData.get(b, h, j, d);
                        }
                        scores.set((float) (score * scale), i, j);
                    }
                }
                
                // 应用因果掩码
                if (useMask) {
                    applyCausalMask(scores, seqLen);
                }
                
                // Softmax
                NdArray attentionWeights = scores.softMax();
                
                // 应用权重到values
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float output = 0.0f;
                        for (int j = 0; j < seqLen; j++) {
                            output += attentionWeights.get(i, j) * valueData.get(b, h, j, d);
                        }
                        attnOutput.set(output, b, h, i, d);
                    }
                }
            }
        }
        
        // 转置回 (batch_size, seq_len, num_heads, head_dim) 并重塑为 (batch_size, seq_len, hidden_size)
        NdArray transposed = transpose4DBack(attnOutput, batchSize, numHeads, seqLen, headDim);
        NdArray concatenated = transposed.reshape(Shape.of(batchSize, seqLen, hiddenSize));
        
        return new Variable(concatenated);
    }
    
    /**
     * 4D张量转置：(batch, heads, seq, dim) -> (batch, seq, heads, dim)
     */
    private NdArray transpose4DBack(NdArray input, int batchSize, int numHeads, int seqLen, int headDim) {
        NdArray output = NdArray.of(Shape.of(batchSize, seqLen, numHeads, headDim));
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    for (int d = 0; d < headDim; d++) {
                        float value = input.get(b, h, s, d);
                        output.set(value, b, s, h, d);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * 应用因果掩码
     */
    private void applyCausalMask(NdArray scores, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                scores.set(Float.NEGATIVE_INFINITY, i, j);
            }
        }
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        Variable[] variables = new Variable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            variables[i] = new Variable(inputs[i]);
        }
        return layerForward(variables).getValue();
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
        return 1; // 基本输入，其他参数可选
    }
    
    /**
     * 获取配置信息
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取层索引
     */
    public int getLayerIdx() {
        return layerIdx;
    }
}