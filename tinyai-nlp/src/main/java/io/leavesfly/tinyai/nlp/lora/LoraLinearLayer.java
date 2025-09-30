package io.leavesfly.tinyai.nlp.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.HashMap;
import java.util.Map;

/**
 * LoRA线性层 - 集成了LoRA适配器的线性层
 * 
 * 该层实现了LoRA微调的核心思想：
 * y = (W + ΔW) * x + b = W * x + ΔW * x + b
 * 其中：
 * - W 是冻结的预训练权重矩阵
 * - ΔW = A * B 是LoRA适配器产生的权重增量
 * - A, B 是可训练的低秩矩阵
 * 
 * 在微调过程中，原始权重W保持冻结，只训练LoRA参数A和B。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class LoraLinearLayer extends Layer {
    
    /**
     * 冻结的预训练权重矩阵
     * 形状: (input_dim, output_dim)
     */
    private final Parameter frozenWeight;
    
    /**
     * 偏置参数（可选）
     * 形状: (1, output_dim)
     */
    private final Parameter bias;
    
    /**
     * LoRA适配器
     */
    private final LoraAdapter loraAdapter;
    
    /**
     * LoRA配置
     */
    private final LoraConfig config;
    
    /**
     * 是否冻结原始权重
     */
    private boolean freezeOriginalWeights = true;
    
    /**
     * 构造函数 - 从头开始创建LoRA线性层
     * 
     * @param _name 层名称
     * @param inputDim 输入维度
     * @param outputDim 输出维度
     * @param config LoRA配置
     * @param needBias 是否需要偏置项
     */
    public LoraLinearLayer(String _name, int inputDim, int outputDim, LoraConfig config, boolean needBias) {
        super(_name, Shape.of(-1, inputDim), Shape.of(-1, outputDim));
        
        this.config = config;
        
        // 验证配置
        config.validate(inputDim, outputDim);
        
        // 初始化原始权重矩阵（可以是预训练的权重）
        NdArray initWeight = NdArray.likeRandomN(Shape.of(inputDim, outputDim))
                .mulNum(Math.sqrt(2.0 / (inputDim + outputDim))); // Xavier初始化
        this.frozenWeight = new Parameter(initWeight);
        this.frozenWeight.setName("frozen_weight");
        this.frozenWeight.setRequireGrad(false); // 冻结权重
        addParam(this.frozenWeight.getName(), this.frozenWeight);
        
        // 初始化偏置项
        if (needBias) {
            this.bias = new Parameter(NdArray.zeros(Shape.of(1, outputDim)));
            this.bias.setName("bias");
            this.bias.setRequireGrad(!config.isEnableBias()); // 根据配置决定是否训练偏置
            addParam(this.bias.getName(), this.bias);
        } else {
            this.bias = null;
        }
        
        // 创建LoRA适配器
        this.loraAdapter = new LoraAdapter(inputDim, outputDim, config);
        
        // 添加LoRA参数到层参数中
        addParam("lora_A", this.loraAdapter.getMatrixA());
        addParam("lora_B", this.loraAdapter.getMatrixB());
    }
    
    /**
     * 构造函数 - 从现有权重创建LoRA层（用于迁移学习）
     * 
     * @param _name 层名称
     * @param pretrainedWeight 预训练权重
     * @param pretrainedBias 预训练偏置（可为null）
     * @param config LoRA配置
     */
    public LoraLinearLayer(String _name, NdArray pretrainedWeight, NdArray pretrainedBias, LoraConfig config) {
        super(_name, Shape.of(-1, pretrainedWeight.getShape().getDimension(0)), 
              Shape.of(-1, pretrainedWeight.getShape().getDimension(1)));
        
        this.config = config;
        
        int inputDim = pretrainedWeight.getShape().getDimension(0);
        int outputDim = pretrainedWeight.getShape().getDimension(1);
        
        // 验证配置
        config.validate(inputDim, outputDim);
        
        // 使用预训练权重并冻结
        this.frozenWeight = new Parameter(pretrainedWeight);
        this.frozenWeight.setName("frozen_weight");
        this.frozenWeight.setRequireGrad(false);
        addParam(this.frozenWeight.getName(), this.frozenWeight);
        
        // 处理偏置项
        if (pretrainedBias != null) {
            this.bias = new Parameter(pretrainedBias);
            this.bias.setName("bias");
            this.bias.setRequireGrad(config.isEnableBias());
            addParam(this.bias.getName(), this.bias);
        } else {
            this.bias = null;
        }
        
        // 创建LoRA适配器
        this.loraAdapter = new LoraAdapter(inputDim, outputDim, config);
        
        // 添加LoRA参数
        addParam("lora_A", this.loraAdapter.getMatrixA());
        addParam("lora_B", this.loraAdapter.getMatrixB());
    }
    
    @Override
    public void init() {
        // 参数已在构造函数中初始化
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        // 1. 计算原始线性变换: input * W_frozen
        Variable originalOutput = input.matMul(frozenWeight);
        
        // 2. 计算LoRA增量: input * A * B * scaling
        Variable loraOutput = loraAdapter.forward(input);
        
        // 3. 合并输出: (W_frozen + ΔW) * input = W_frozen * input + ΔW * input
        Variable combinedOutput = originalOutput.add(loraOutput);
        
        // 4. 添加偏置项
        if (bias != null) {
            combinedOutput = combinedOutput.add(bias);
        }
        
        return combinedOutput;
    }
    
    /**
     * 启用LoRA适配器
     */
    public void enableLora() {
        loraAdapter.enable();
    }
    
    /**
     * 禁用LoRA适配器（仅使用原始权重）
     */
    public void disableLora() {
        loraAdapter.disable();
    }
    
    /**
     * 检查LoRA是否启用
     * 
     * @return LoRA是否启用
     */
    public boolean isLoraEnabled() {
        return loraAdapter.isEnabled();
    }
    
    /**
     * 冻结原始权重
     */
    public void freezeOriginalWeights() {
        this.freezeOriginalWeights = true;
        frozenWeight.setRequireGrad(false);
    }
    
    /**
     * 解冻原始权重（允许全参数微调）
     */
    public void unfreezeOriginalWeights() {
        this.freezeOriginalWeights = false;
        frozenWeight.setRequireGrad(true);
    }
    
    /**
     * 检查原始权重是否冻结
     * 
     * @return 原始权重是否冻结
     */
    public boolean isOriginalWeightsFrozen() {
        return freezeOriginalWeights;
    }
    
    /**
     * 合并LoRA权重到原始权重中
     * 这个操作会将 W = W + A * B * scaling
     * 合并后可以移除LoRA参数，获得等效的单一权重矩阵
     * 
     * @return 合并后的权重矩阵
     */
    public NdArray mergeLoraWeights() {
        if (!loraAdapter.isEnabled()) {
            return frozenWeight.getValue();
        }
        
        // 计算 ΔW = A * B * scaling
        Variable deltaW = loraAdapter.getMatrixA().matMul(loraAdapter.getMatrixB());
        if (loraAdapter.getScaling() != 1.0) {
            Variable scalingVar = new Variable(NdArray.of(loraAdapter.getScaling()));
            deltaW = deltaW.mul(scalingVar);
        }
        
        // 合并权重: W_new = W_frozen + ΔW
        return frozenWeight.getValue().add(deltaW.getValue());
    }
    
    /**
     * 获取可训练参数数量
     * 
     * @return 可训练参数数量
     */
    public int getTrainableParameterCount() {
        int count = loraAdapter.getParameterCount();
        
        if (bias != null && bias.isRequireGrad()) {
            count += bias.getValue().getShape().size();
        }
        
        if (!freezeOriginalWeights) {
            count += frozenWeight.getValue().getShape().size();
        }
        
        return count;
    }
    
    /**
     * 获取总参数数量
     * 
     * @return 总参数数量
     */
    public int getTotalParameterCount() {
        int count = frozenWeight.getValue().getShape().size() + loraAdapter.getParameterCount();
        
        if (bias != null) {
            count += bias.getValue().getShape().size();
        }
        
        return count;
    }
    
    /**
     * 获取相对于全参数微调的参数减少比例
     * 
     * @return 参数减少比例
     */
    public double getParameterReduction() {
        int totalParams = getTotalParameterCount();
        int trainableParams = getTrainableParameterCount();
        
        // 计算相对于全参数微调的减少比例
        return 1.0 - (double) trainableParams / totalParams;
    }
    
    @Override
    public void clearGrads() {
        super.clearGrads();
        loraAdapter.clearGrads();
    }
    
    /**
     * 获取LoRA适配器
     * 
     * @return LoRA适配器
     */
    public LoraAdapter getLoraAdapter() {
        return loraAdapter;
    }
    
    /**
     * 获取LoRA配置
     * 
     * @return LoRA配置
     */
    public LoraConfig getLoraConfig() {
        return config;
    }
    
    /**
     * 获取冻结权重
     * 
     * @return 冻结权重参数
     */
    public Parameter getFrozenWeight() {
        return frozenWeight;
    }
    
    /**
     * 获取偏置参数
     * 
     * @return 偏置参数（可能为null）
     */
    public Parameter getBias() {
        return bias;
    }
    
    /**
     * 获取所有LoRA相关的参数
     * 
     * @return LoRA参数映射
     */
    public Map<String, Parameter> getLoraParameters() {
        Map<String, Parameter> loraParams = new HashMap<>();
        loraParams.put(getName() + ".lora_A", loraAdapter.getMatrixA());
        loraParams.put(getName() + ".lora_B", loraAdapter.getMatrixB());
        return loraParams;
    }
    
    @Override
    public String toString() {
        return String.format(
            "LoraLinearLayer{name='%s', inputDim=%d, outputDim=%d, config=%s, trainableParams=%d/%d (%.1f%% reduction)}", 
            getName(),
            getInputShape().getDimension(1), 
            getOutputShape().getDimension(1),
            config.toString(),
            getTrainableParameterCount(),
            getTotalParameterCount(),
            getParameterReduction() * 100
        );
    }
}