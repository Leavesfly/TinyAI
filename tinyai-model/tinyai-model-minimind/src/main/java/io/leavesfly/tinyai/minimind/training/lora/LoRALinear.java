package io.leavesfly.tinyai.minimind.training.lora;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.init.Initializers;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;

/**
 * LoRA线性层
 * 
 * LoRA (Low-Rank Adaptation)通过低秩分解实现参数高效微调
 * 原理: W' = W + (α/r) * A * B
 * 其中:
 * - W是原始权重(冻结)
 * - A是低秩矩阵(r × in_features)
 * - B是低秩矩阵(out_features × r)
 * - r是LoRA秩
 * - α是缩放因子
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoRALinear extends Module {
    
    // 原始Linear层的权重(冻结)
    private final Parameter originalWeight;
    private final Parameter originalBias;
    
    // LoRA参数(可训练)
    private Parameter loraA;  // shape: (r, in_features)
    private Parameter loraB;  // shape: (out_features, r)
    
    // LoRA配置
    public final int inFeatures;
    public final int outFeatures;
    private final int rank;
    private final float alpha;
    private final float scaling;  // scaling = alpha / rank
    private final boolean useBias;
    
    // Dropout层(可选)
    private Dropout dropout;
    
    /**
     * 构造函数
     * 
     * @param name 层名称
     * @param inFeatures 输入特征数
     * @param outFeatures 输出特征数
     * @param useBias 是否使用偏置
     * @param rank LoRA秩
     * @param alpha LoRA缩放因子
     * @param dropoutRate LoRA dropout比例
     */
    public LoRALinear(String name, int inFeatures, int outFeatures, boolean useBias,
                      int rank, float alpha, float dropoutRate) {
        super(name);
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.rank = rank;
        this.alpha = alpha;
        this.scaling = alpha / rank;
        this.useBias = useBias;
        
        // 创建原始权重(冻结,不参与训练)
        NdArray weightData = NdArray.of(Shape.of(outFeatures, inFeatures));
        this.originalWeight = new Parameter(weightData);
        
        if (useBias) {
            NdArray biasData = NdArray.of(Shape.of(outFeatures));
            this.originalBias = new Parameter(biasData);
        } else {
            this.originalBias = null;
        }
        
        // 创建LoRA参数(可训练)
        NdArray loraAData = NdArray.of(Shape.of(rank, inFeatures));
        NdArray loraBData = NdArray.of(Shape.of(outFeatures, rank));
        
        this.loraA = registerParameter("lora_A", new Parameter(loraAData));
        this.loraB = registerParameter("lora_B", new Parameter(loraBData));
        
        // 创建Dropout层
        if (dropoutRate > 0) {
            this.dropout = new Dropout("lora_dropout", dropoutRate);
            registerModule("lora_dropout", dropout);
        }
        
        // 初始化参数
        init();
    }
    
    /**
     * 简化构造函数(默认dropout=0.1)
     */
    public LoRALinear(String name, int inFeatures, int outFeatures, boolean useBias,
                      int rank, float alpha) {
        this(name, inFeatures, outFeatures, useBias, rank, alpha, 0.1f);
    }
    
    /**
     * 从现有Linear层创建LoRA层
     * 
     * @param name 层名称
     * @param originalWeight 原始权重
     * @param originalBias 原始偏置(可为null)
     * @param rank LoRA秩
     * @param alpha LoRA缩放因子
     * @param dropoutRate Dropout比例
     */
    public static LoRALinear fromLinear(String name, Parameter originalWeight, 
                                        Parameter originalBias,
                                        int rank, float alpha, float dropoutRate) {
        int[] weightShape = originalWeight.data().getShape().getShapeDims();
        int outFeatures = weightShape[0];
        int inFeatures = weightShape[1];
        boolean useBias = originalBias != null;
        
        LoRALinear loraLinear = new LoRALinear(name, inFeatures, outFeatures, 
                                               useBias, rank, alpha, dropoutRate);
        
        // 复制原始权重
        loraLinear.setOriginalWeight(originalWeight.data());
        if (useBias && originalBias != null) {
            loraLinear.setOriginalBias(originalBias.data());
        }
        
        return loraLinear;
    }
    
    /**
     * 设置原始权重
     */
    public void setOriginalWeight(NdArray weight) {
        // 复制权重数据
        float[] src = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) weight).buffer;
        float[] dst = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) this.originalWeight.data()).buffer;
        System.arraycopy(src, 0, dst, 0, src.length);
    }
    
    /**
     * 设置原始偏置
     */
    public void setOriginalBias(NdArray bias) {
        if (this.originalBias != null && bias != null) {
            // 复制偏置数据
            float[] src = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) bias).buffer;
            float[] dst = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) this.originalBias.data()).buffer;
            System.arraycopy(src, 0, dst, 0, src.length);
        }
    }
    
    @Override
    public void resetParameters() {
        // 初始化原始权重(Kaiming初始化)
        Initializers.kaimingUniform(originalWeight.data(), 0, "fan_in", "relu");
        if (originalBias != null) {
            Initializers.zeros(originalBias.data());
        }
        
        // LoRA A使用Kaiming初始化
        Initializers.kaimingUniform(loraA.data(), 0, "fan_in", "relu");
        
        // LoRA B初始化为0,确保初始时LoRA不影响原始输出
        Initializers.zeros(loraB.data());
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];  // shape: (batch, in_features)
        
        // 1. 原始线性变换: y = xW^T
        // originalWeight.shape: (out_features, in_features)
        Variable y = x.matMul(transposeWeight(originalWeight.data()));
        
        // 2. 添加原始偏置
        if (originalBias != null) {
            y = y.add(new Variable(originalBias.data()));
        }
        
        // 3. LoRA低秩调整: delta = x * A^T * B^T * (α/r)
        // Step 3.1: x * A^T -> (batch, r)
        Variable loraX = x.matMul(transposeWeight(loraA.data()));
        
        // Step 3.2: 应用Dropout(训练模式)
        if (dropout != null && _training) {
            loraX = dropout.forward(loraX);
        }
        
        // Step 3.3: (x * A^T) * B^T -> (batch, out_features)
        Variable loraDelta = loraX.matMul(transposeWeight(loraB.data()));
        
        // Step 3.4: 应用缩放因子
        loraDelta = loraDelta.mul(new Variable(NdArray.of(scaling)));
        
        // 4. 合并: y = y_orig + lora_delta
        y = y.add(loraDelta);
        
        return y;
    }
    
    /**
     * 转置权重矩阵
     */
    private Variable transposeWeight(NdArray weight) {
        return new Variable(weight).transpose();
    }
    
    /**
     * 合并LoRA权重到原始权重
     * 
     * 合并后可以去掉LoRA层,直接使用合并后的权重进行推理
     * W_merged = W + (α/r) * B * A
     * 
     * @return 合并后的权重
     */
    public NdArray mergeWeights() {
        // 计算 B * A (使用Variable进行矩阵乘法)
        Variable loraBVar = new Variable(loraB.data());
        Variable loraAVar = new Variable(loraA.data());
        Variable loraBAVar = loraBVar.matMul(loraAVar);
        
        // 应用缩放因子: (α/r) * B * A
        NdArray scaledLoRA = loraBAVar.getValue().mulNum(scaling);
        
        // 合并: W + scaled_lora
        NdArray mergedWeight = originalWeight.data().add(scaledLoRA);
        
        return mergedWeight;
    }
    
    /**
     * 获取LoRA参数量
     */
    public int getLoRAParams() {
        return (rank * inFeatures) + (outFeatures * rank);
    }
    
    /**
     * 获取原始参数量
     */
    public int getOriginalParams() {
        int params = outFeatures * inFeatures;
        if (useBias) {
            params += outFeatures;
        }
        return params;
    }
    
    /**
     * 获取参数压缩比
     */
    public float getCompressionRatio() {
        return (float) getLoRAParams() / getOriginalParams() * 100;
    }
    
    /**
     * 打印LoRA信息
     */
    public void printLoRAInfo() {
        System.out.println("=".repeat(60));
        System.out.println("LoRA Linear层信息:");
        System.out.println("  层名称: " + name);
        System.out.println("  输入维度: " + inFeatures);
        System.out.println("  输出维度: " + outFeatures);
        System.out.println("  LoRA秩: " + rank);
        System.out.println("  缩放因子: " + alpha);
        System.out.println("  缩放比例: " + scaling);
        System.out.println("  原始参数量: " + getOriginalParams());
        System.out.println("  LoRA参数量: " + getLoRAParams());
        System.out.println("  参数压缩比: " + String.format("%.2f%%", getCompressionRatio()));
        System.out.println("=".repeat(60));
    }
    
    // Getters
    
    public Parameter getOriginalWeight() {
        return originalWeight;
    }
    
    public Parameter getOriginalBias() {
        return originalBias;
    }
    
    public Parameter getLoraA() {
        return loraA;
    }
    
    public Parameter getLoraB() {
        return loraB;
    }
    
    public int getRank() {
        return rank;
    }
    
    public float getAlpha() {
        return alpha;
    }
    
    public float getScaling() {
        return scaling;
    }
    
    @Override
    public String toString() {
        return String.format(
            "LoRALinear{name='%s', in=%d, out=%d, rank=%d, alpha=%.1f, scaling=%.3f, params=%d/%d (%.2f%%)}",
            name, inFeatures, outFeatures, rank, alpha, scaling, 
            getLoRAParams(), getOriginalParams(), getCompressionRatio()
        );
    }
}
