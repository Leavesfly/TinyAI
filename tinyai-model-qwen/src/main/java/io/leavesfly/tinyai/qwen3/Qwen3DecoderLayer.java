package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3解码器层
 * 
 * 实现了Qwen3模型的单个Transformer解码器层，包含以下组件：
 * 1. 多头自注意力机制（带RoPE和GQA）
 * 2. 前归一化（Pre-LayerNorm）
 * 3. SwiGLU前馈网络
 * 4. 残差连接
 * 
 * 层结构（Pre-LN）：
 * x -> LayerNorm -> SelfAttention -> Add -> LayerNorm -> MLP -> Add
 * |                                  ^                        ^
 * |__________________________________|________________________|
 * 
 * 这种Pre-LN结构在训练大型语言模型时更加稳定。
 * 
 * @author 山泽
 * @version 1.0
 */
public class Qwen3DecoderLayer extends Layer {
    
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
     * 自注意力层
     */
    private Qwen3Attention selfAttn;
    
    /**
     * 前馈网络层
     */
    private Qwen3MLP mlp;
    
    /**
     * 注意力前的层归一化
     */
    private RMSNorm inputLayerNorm;
    
    /**
     * MLP前的层归一化
     */
    private RMSNorm postAttentionLayerNorm;

    /**
     * 构造Qwen3解码器层
     * 
     * @param name 层名称
     * @param config 配置信息
     * @param layerIdx 层索引
     */
    public Qwen3DecoderLayer(String name, Qwen3Config config, int layerIdx) {
        super(name, Shape.of(-1, -1, config.getHiddenSize()), Shape.of(-1, -1, config.getHiddenSize()));
        
        this.config = config;
        this.layerIdx = layerIdx;
        this.hiddenSize = config.getHiddenSize();
        
        init();
    }

    @Override
    public void init() {
        if (!alreadyInit) {
            // 初始化自注意力层
            selfAttn = new Qwen3Attention(name + "_self_attn", config, layerIdx, true);
            
            // 初始化前馈网络
            mlp = new Qwen3MLP(name + "_mlp", config);
            
            // 初始化层归一化
            inputLayerNorm = new RMSNorm(name + "_input_layernorm", hiddenSize, config.getRmsNormEps());
            postAttentionLayerNorm = new RMSNorm(name + "_post_attention_layernorm", hiddenSize, config.getRmsNormEps());
            
            alreadyInit = true;
        }
    }

    @Override
    public Variable layerForward(Variable... inputs) {
        Variable hiddenStates = inputs[0];
        // 其他可选输入（注意力掩码、位置ID等）暂时忽略
        
        Variable residual = hiddenStates;
        
        // 第一个残差分支：Pre-LN + 自注意力
        // 1. 前归一化
        Variable normalizedInput = inputLayerNorm.layerForward(hiddenStates);
        
        // 2. 自注意力
        Variable attnOutput = selfAttn.layerForward(normalizedInput);
        
        // 3. 残差连接
        Variable afterFirstResidual = addResidualConnection(residual, attnOutput);
        
        // 第二个残差分支：Pre-LN + MLP
        residual = afterFirstResidual;
        
        // 4. 前归一化
        Variable normalizedResidual = postAttentionLayerNorm.layerForward(afterFirstResidual);
        
        // 5. MLP
        Variable mlpOutput = mlp.layerForward(normalizedResidual);
        
        // 6. 残差连接
        Variable finalOutput = addResidualConnection(residual, mlpOutput);
        
        return finalOutput;
    }
    
    /**
     * 执行残差连接：output = input + residual
     * 
     * @param residual 残差连接的输入
     * @param output 当前层的输出
     * @return 残差连接后的结果
     */
    private Variable addResidualConnection(Variable residual, Variable output) {
        NdArray residualData = residual.getValue();
        NdArray outputData = output.getValue();
        
        // 验证形状匹配
        if (!residualData.getShape().equals(outputData.getShape())) {
            throw new IllegalArgumentException(
                "残差连接形状不匹配: residual=" + residualData.getShape() + 
                ", output=" + outputData.getShape()
            );
        }
        
        // 执行逐元素相加
        NdArray result = addElementwise(residualData, outputData);
        return new Variable(result);
    }
    
    /**
     * 逐元素相加操作
     */
    private NdArray addElementwise(NdArray a, NdArray b) {
        Shape shape = a.getShape();
        NdArray result = NdArray.of(shape);
        
        if (shape.getDimNum() == 2) {
            // 2D情况: (batch_size, hidden_size)
            int batchSize = shape.getDimension(0);
            int hiddenSize = shape.getDimension(1);
            
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    float sum = a.get(i, j) + b.get(i, j);
                    result.set(sum, i, j);
                }
            }
        } else if (shape.getDimNum() == 3) {
            // 3D情况: (batch_size, seq_len, hidden_size)
            int batchSize = shape.getDimension(0);
            int seqLen = shape.getDimension(1);
            int hiddenSize = shape.getDimension(2);
            
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < hiddenSize; k++) {
                        float sum = a.get(i, j, k) + b.get(i, j, k);
                        result.set(sum, i, j, k);
                    }
                }
            }
        } else {
            throw new IllegalArgumentException("不支持的张量维度: " + shape.getDimNum());
        }
        
        return result;
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
        // 解码器层的反向传播需要通过所有子层
        // 简化实现返回原梯度
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
     * 
     * @return 配置信息
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取层索引
     * 
     * @return 层索引
     */
    public int getLayerIdx() {
        return layerIdx;
    }
    
    /**
     * 获取隐藏层维度
     * 
     * @return 隐藏层维度
     */
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    /**
     * 获取自注意力层
     * 
     * @return 自注意力层
     */
    public Qwen3Attention getSelfAttn() {
        return selfAttn;
    }
    
    /**
     * 获取前馈网络层
     * 
     * @return 前馈网络层
     */
    public Qwen3MLP getMlp() {
        return mlp;
    }
    
    /**
     * 获取输入层归一化
     * 
     * @return 输入层归一化
     */
    public RMSNorm getInputLayerNorm() {
        return inputLayerNorm;
    }
    
    /**
     * 获取注意力后层归一化
     * 
     * @return 注意力后层归一化
     */
    public RMSNorm getPostAttentionLayerNorm() {
        return postAttentionLayerNorm;
    }
}