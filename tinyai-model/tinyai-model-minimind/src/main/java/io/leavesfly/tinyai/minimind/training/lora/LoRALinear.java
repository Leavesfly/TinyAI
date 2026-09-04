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

    /**
     * 原始权重是否已由外部显式写入
     * <p>
     * 用于保护 {@link #resetParameters()}：一旦通过 {@link #setOriginalWeight(NdArray)}
     * 载入了预训练权重，后续任何 init()/resetParameters() 调用都不得再将其随机化，
     * 否则会毁掉 LoRA 注入前训练好的基座权重。
     */
    private boolean originalWeightLoaded = false;

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
        // requireGrad=false：反向传播到此叶子即终止，既不会被优化器更新，也不浪费算力累积梯度
        NdArray weightData = NdArray.of(Shape.of(outFeatures, inFeatures));
        this.originalWeight = new Parameter(weightData, false);
        
        if (useBias) {
            NdArray biasData = NdArray.of(Shape.of(outFeatures));
            this.originalBias = new Parameter(biasData, false);
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
        // 复制权重数据（仅用 NdArray 公开接口，不假设 CPU 后端）
        this.originalWeight.setValue(weight.copy());
        this.originalWeightLoaded = true;
    }
    
    /**
     * 设置原始偏置
     */
    public void setOriginalBias(NdArray bias) {
        if (this.originalBias != null && bias != null) {
            // 复制偏置数据（仅用 NdArray 公开接口，不假设 CPU 后端）
            this.originalBias.setValue(bias.copy());
        }
    }
    
    @Override
    public void resetParameters() {
        // 初始化原始权重(Kaiming初始化)
        // 已由外部载入预训练权重时跳过，避免 init() 被重复调用时毁掉基座权重
        if (!originalWeightLoaded) {
            Initializers.kaimingUniform(originalWeight.data(), 0, "fan_in", "relu");
        }
        if (originalBias != null && !originalWeightLoaded) {
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
        // 直接使用 Parameter 本身参与计算图（Parameter extends Variable）。
        // 该参数 requireGrad=false，因此反向传播到此终止，实现真正的"冻结"。
        // originalWeight.shape: (out_features, in_features)
        Variable y = x.matMul(originalWeight.transpose());
        
        // 2. 添加原始偏置
        if (originalBias != null) {
            y = y.add(originalBias);
        }
        
        // 3. LoRA低秩调整: delta = dropout(x) * A^T * B^T * (α/r)
        // 关键：loraA / loraB 必须以 Parameter 对象本身进入计算图。
        // 若写成 new Variable(loraA.data())，图里的叶子会是这个临时 Variable，
        // 梯度落在临时对象上被丢弃，loraA.getGrad() 恒为 null，优化器跳过 null 梯度
        // → LoRA 参数永远不更新，训练完全空转。
        // Step 3.1: 对输入应用 Dropout(训练模式)，对标标准 LoRA: lora_B(lora_A(dropout(x)))
        Variable loraInput = (dropout != null && _training) ? dropout.forward(x) : x;
        
        // Step 3.2: dropout(x) * A^T -> (batch, r)
        Variable loraX = loraInput.matMul(loraA.transpose());
        
        // Step 3.3: (x * A^T) * B^T -> (batch, out_features)
        Variable loraDelta = loraX.matMul(loraB.transpose());
        
        // Step 3.4: 应用缩放因子
        Variable scalingVar = new Variable(NdArray.of(scaling));
        scalingVar.setRequireGrad(false);
        loraDelta = loraDelta.mul(scalingVar);
        
        // 4. 合并: y = y_orig + lora_delta
        y = y.add(loraDelta);
        
        return y;
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
