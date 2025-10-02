package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.nnet.layer.transformer.MultiHeadAttention;

/**
 * DeepSeek V3增强的Transformer块
 * 
 * 相比标准Transformer块，V3TransformerBlock包含以下增强特性：
 * 1. 混合专家模型(MoE)前馈网络 - 替代传统的FFN
 * 2. 门控机制 - 控制MoE输出与残差连接的平衡
 * 3. 任务类型感知 - 根据任务类型调整专家选择
 * 4. 增强的层归一化 - 更好的训练稳定性
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3TransformerBlock extends Block {
    
    /**
     * 模型维度
     */
    private final int dModel;
    
    /**
     * 注意力头数
     */
    private final int numHeads;
    
    /**
     * 前馈网络维度
     */
    private final int dFF;
    
    /**
     * 专家数量
     */
    private final int numExperts;
    
    /**
     * Dropout概率
     */
    private final float dropout;
    
    /**
     * 多头注意力层
     */
    private MultiHeadAttention attention;
    
    /**
     * 第一个层归一化（注意力前）
     */
    private LayerNorm norm1;
    
    /**
     * MoE前馈网络
     */
    private MixtureOfExperts moeFFN;
    
    /**
     * 第二个层归一化（MoE前）
     */
    private LayerNorm norm2;
    
    /**
     * 门控机制 - 控制MoE输出的权重
     */
    private LinearLayer gate;
    
    /**
     * 最后一次前向传播的MoE路由信息
     */
    private ExpertRoutingInfo lastRoutingInfo;
    
    /**
     * 构造函数
     * 
     * @param name Transformer块名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dFF 前馈网络维度
     * @param numExperts 专家数量
     * @param dropout Dropout概率
     */
    public V3TransformerBlock(String name, int dModel, int numHeads, int dFF, 
                             int numExperts, float dropout) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.numExperts = numExperts;
        this.dropout = dropout;
        
        init();
    }
    
    /**
     * 默认构造函数 - 使用标准配置
     */
    public V3TransformerBlock(String name, int dModel, int numHeads) {
        this(name, dModel, numHeads, dModel * 4, 8, 0.1f);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 多头注意力层
            attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, false);
            addLayer(attention);
            
            // 第一个层归一化
            norm1 = new LayerNorm(name + "_norm1", dModel);
            addLayer(norm1);
            
            // MoE前馈网络
            moeFFN = new MixtureOfExperts(name + "_moe_ffn", dModel, numExperts, 2, 1.0f);
            addLayer(moeFFN);
            
            // 第二个层归一化
            norm2 = new LayerNorm(name + "_norm2", dModel);
            addLayer(norm2);
            
            // 门控机制 - 单个输出用于控制MoE权重
            gate = new LinearLayer(name + "_gate", dModel, 1, true);
            addLayer(gate);
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        // 默认任务类型
        TaskType taskType = TaskType.GENERAL;
        
        // 如果提供了任务类型参数（可扩展）
        if (inputs.length > 1) {
            // 这里可以扩展任务类型传递机制
            taskType = TaskType.GENERAL;
        }
        
        return forwardWithTaskType(input, null, taskType);
    }
    
    /**
     * 执行带任务类型和掩码的前向传播
     * 
     * @param x 输入变量
     * @param mask 注意力掩码（可为null）
     * @param taskType 任务类型
     * @return 输出变量
     */
    public Variable forwardWithTaskType(Variable x, NdArray mask, TaskType taskType) {
        // 保存原始输入
        Variable originalInput = x;
        
        // 自注意力机制
        // 先进行层归一化（Pre-LN架构）
        Variable normed1 = norm1.layerForward(x);
        
        // 多头自注意力
        Variable attended = attention.layerForward(normed1, normed1, normed1);
        
        // 残差连接
        Variable afterAttention = addResidual(x, attended);
        
        // MoE前馈网络
        // 第二次层归一化
        Variable normed2 = norm2.layerForward(afterAttention);
        
        // MoE前向传播
        MixtureOfExperts.MoEResult moeResult = moeFFN.forwardWithTaskType(normed2, taskType);
        Variable moeOutput = moeResult.output;
        
        // 保存路由信息
        lastRoutingInfo = moeResult.routingInfo;
        
        // 门控机制
        Variable gateInput = afterAttention; // 使用注意力后的特征计算门控
        Variable gateLogits = gate.layerForward(gateInput);
        NdArray gateWeights = applySigmoid(gateLogits.getValue());
        
        // 应用门控权重
        Variable gatedOutput = applyGating(moeOutput, afterAttention, gateWeights);
        
        // 最终残差连接
        Variable finalOutput = addResidual(afterAttention, gatedOutput);
        
        return finalOutput;
    }
    
    /**
     * 添加残差连接
     */
    private Variable addResidual(Variable residual, Variable main) {
        NdArray residualData = residual.getValue();
        NdArray mainData = main.getValue();
        NdArray result = residualData.add(mainData);
        return new Variable(result);
    }
    
    /**
     * 从线性索引计算多维索引
     */
    private int[] getIndicesFromLinearIndex(int linearIndex, Shape shape) {
        if (shape.isMatrix()) {
            int row = linearIndex / shape.getColumn();
            int col = linearIndex % shape.getColumn();
            return new int[]{row, col};
        } else if (shape.getDimNum() == 3) {
            int dim1 = shape.getDimension(1);
            int dim2 = shape.getDimension(2);
            int area = dim1 * dim2;
            int d0 = linearIndex / area;
            int remainder = linearIndex % area;
            int d1 = remainder / dim2;
            int d2 = remainder % dim2;
            return new int[]{d0, d1, d2};
        } else {
            // 默认一维索引
            return new int[]{linearIndex};
        }
    }
    
    /**
     * 应用Sigmoid激活函数
     */
    private NdArray applySigmoid(NdArray input) {
        Shape shape = input.getShape();
        NdArray result = NdArray.of(shape);
        
        for (int i = 0; i < shape.size(); i++) {
            // 计算多维索引
            int[] indices = getIndicesFromLinearIndex(i, shape);
            float value = input.get(indices);
            float sigmoid = 1.0f / (1.0f + (float)Math.exp(-value));
            result.set(sigmoid, indices);
        }
        
        return result;
    }
    
    /**
     * 应用门控机制
     * 
     * @param moeOutput MoE输出
     * @param residualInput 残差输入
     * @param gateWeights 门控权重
     * @return 门控后的输出
     */
    private Variable applyGating(Variable moeOutput, Variable residualInput, NdArray gateWeights) {
        NdArray moeData = moeOutput.getValue();
        NdArray residualData = residualInput.getValue();
        
        int batchSize = moeData.getShape().getDimension(0);
        int seqLen = moeData.getShape().getDimension(1);
        int dModel = moeData.getShape().getDimension(2);
        
        NdArray result = NdArray.of(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 获取该位置的门控权重
                float gateWeight = gateWeights.get(b, s, 0);
                
                for (int d = 0; d < dModel; d++) {
                    float moeValue = moeData.get(b, s, d);
                    float residualValue = residualData.get(b, s, d);
                    
                    // 门控组合：gate_weight * moe_output + (1 - gate_weight) * residual
                    float gatedValue = gateWeight * moeValue + (1 - gateWeight) * residualValue;
                    result.set(gatedValue, b, s, d);
                }
            }
        }
        
        return new Variable(result);
    }
    
    /**
     * 获取最后一次前向传播的MoE路由信息
     */
    public ExpertRoutingInfo getLastRoutingInfo() {
        return lastRoutingInfo;
    }
    
    /**
     * 获取MoE损失（用于训练时的损失计算）
     */
    public float getMoELoss() {
        if (lastRoutingInfo != null) {
            return lastRoutingInfo.getTotalMoELoss();
        }
        return 0.0f;
    }
    
    /**
     * 重置路由信息
     */
    public void resetRoutingInfo() {
        lastRoutingInfo = null;
    }
    
    // Getters
    public int getDModel() {
        return dModel;
    }
    
    public int getNumHeads() {
        return numHeads;
    }
    
    public int getDFF() {
        return dFF;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public float getDropout() {
        return dropout;
    }
    
    public MultiHeadAttention getAttention() {
        return attention;
    }
    
    public MixtureOfExperts getMoeFFN() {
        return moeFFN;
    }
    
    public LayerNorm getNorm1() {
        return norm1;
    }
    
    public LayerNorm getNorm2() {
        return norm2;
    }
    
    public LinearLayer getGate() {
        return gate;
    }
    
    @Override
    public String toString() {
        return String.format("V3TransformerBlock{name='%s', dModel=%d, numHeads=%d, numExperts=%d}", 
                           name, dModel, numHeads, numExperts);
    }
}