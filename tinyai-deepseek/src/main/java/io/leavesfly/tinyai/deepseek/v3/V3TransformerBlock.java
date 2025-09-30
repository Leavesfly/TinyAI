package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Tanh;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.layer.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

/**
 * DeepSeek V3 增强的Transformer块
 * 
 * 集成了以下V3特性：
 * 1. 多头注意力机制
 * 2. 混合专家模型(MoE)前馈网络
 * 3. 门控机制用于控制MoE输出
 * 4. 残差连接和层归一化
 * 5. 任务类型感知处理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class V3TransformerBlock extends LayerAble {
    
    // ========== 网络组件 ==========
    private MultiHeadAttention attention;     // 多头注意力
    private LayerNorm norm1;                  // 第一个层归一化
    private MixtureOfExperts moeFFN;          // MoE前馈网络
    private LayerNorm norm2;                  // 第二个层归一化
    private Linear gateLayer;                 // 门控层
    
    // ========== 配置参数 ==========
    private int dModel;                       // 模型维度
    private int numHeads;                     // 注意力头数
    private int dFF;                          // 前馈网络维度
    private int numExperts;                   // 专家数量
    private double dropout;                   // dropout比率
    
    // ========== 运行时状态 ==========
    private ExpertRoutingInfo lastRoutingInfo; // 最近的路由信息
    private TaskType currentTaskType;           // 当前处理的任务类型
    
    /**
     * 构造函数
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param dFF 前馈网络维度
     * @param numExperts 专家数量
     * @param dropout dropout比率
     */
    public V3TransformerBlock(String name, int dModel, int numHeads, int dFF, 
                             int numExperts, double dropout) {
        this.name = name;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.numExperts = numExperts;
        this.dropout = dropout;
        
        // 设置输入输出形状
        this.inputShape = Shape.of(-1, -1, dModel);  // [batch, seq, dModel]
        this.outputShape = Shape.of(-1, -1, dModel); // [batch, seq, dModel]
        
        init();
    }
    
    /**
     * 简化构造函数
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numHeads 注意力头数
     * @param numExperts 专家数量
     */
    public V3TransformerBlock(String name, int dModel, int numHeads, int numExperts) {
        this(name, dModel, numHeads, dModel * 4, numExperts, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化多头注意力
            attention = new MultiHeadAttention(name + "_attention", dModel, numHeads, dropout);
            
            // 2. 初始化层归一化
            norm1 = new LayerNorm(name + "_norm1", dModel);
            norm2 = new LayerNorm(name + "_norm2", dModel);
            
            // 3. 初始化MoE前馈网络
            moeFFN = new MixtureOfExperts(name + "_moe", dModel, numExperts, 2, 1.0);
            
            // 4. 初始化门控层
            gateLayer = new Linear(name + "_gate", dModel, 1, false);
            
            // 初始化所有组件
            attention.init();
            norm1.init();
            norm2.init();
            moeFFN.init();
            gateLayer.init();
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable x = inputs[0];  // [batch_size, seq_len, d_model]
        Variable attentionMask = inputs.length > 1 ? inputs[1] : null;
        
        // 提取任务类型信息（如果有）
        TaskType taskType = extractTaskType(inputs);
        this.currentTaskType = taskType;
        
        // 1. 多头自注意力 + 残差连接
        Variable attended = attention.forward(x, x, x, attentionMask);
        x = norm1.layerForward(x.add(attended));
        
        // 2. MoE前馈网络
        Variable moeInput = x;
        Variable moeOutput;
        
        if (taskType != null) {
            // 传递任务类型信息给MoE
            moeOutput = moeFFN.layerForward(moeInput, createTaskTypeVariable(taskType));
        } else {
            moeOutput = moeFFN.layerForward(moeInput);
        }
        
        // 保存路由信息（简化实现）
        this.lastRoutingInfo = createMockRoutingInfo();
        
        // 3. 门控机制
        Variable gateWeights = gateLayer.layerForward(x);
        gateWeights = applySigmoid(gateWeights); // Sigmoid激活
        
        Variable gatedOutput = applyGating(x, moeOutput, gateWeights);
        
        // 4. 第二个残差连接和层归一化
        x = norm2.layerForward(x.add(gatedOutput));
        
        return x;
    }
    
    /**
     * 从输入中提取任务类型
     * 简化实现，实际可以通过更复杂的方式推断
     */
    private TaskType extractTaskType(Variable[] inputs) {
        // 如果输入包含任务类型标识，解析它
        if (inputs.length > 2 && inputs[2] != null) {
            // 这里可以实现更复杂的任务类型推断逻辑
            // 简化实现：返回默认任务类型
            return TaskType.GENERAL;
        }
        return null;
    }
    
    /**
     * 创建任务类型变量
     * 用于传递给MoE层
     */
    private Variable createTaskTypeVariable(TaskType taskType) {
        // 创建一个简单的任务类型编码
        double[] encoding = new double[TaskType.values().length];
        encoding[taskType.ordinal()] = 1.0;
        return new Variable(NdArray.of(encoding).reshape(Shape.of(1, TaskType.values().length)));
    }
    
    /**
     * 创建模拟的路由信息
     * 实际实现中应该从MoE层获取真实的路由信息
     */
    private ExpertRoutingInfo createMockRoutingInfo() {
        double[] weights = new double[numExperts];
        java.util.List<Integer> selected = new java.util.ArrayList<>();
        
        // 简化的模拟数据
        weights[0] = 0.6;
        weights[1] = 0.4;
        selected.add(0);
        selected.add(1);
        
        return new ExpertRoutingInfo(weights, selected, 0.1, 0.05);
    }
    
    /**
     * 应用Sigmoid激活函数
     */
    private Variable applySigmoid(Variable x) {
        // 手动实现Sigmoid: 1 / (1 + exp(-x))
        NdArray data = x.getValue();
        double[] array = data.toDoubleArray();
        
        for (int i = 0; i < array.length; i++) {
            array[i] = 1.0 / (1.0 + Math.exp(-array[i]));
        }
        
        return new Variable(NdArray.of(array).reshape(data.getShape()));
    }
    
    /**
     * 应用门控机制
     * 
     * @param original 原始输入
     * @param moeOutput MoE输出
     * @param gateWeights 门控权重
     * @return 门控后的输出
     */
    private Variable applyGating(Variable original, Variable moeOutput, Variable gateWeights) {
        // gatedOutput = gateWeight * moeOutput + (1 - gateWeight) * original
        
        // 扩展门控权重到匹配输出维度
        Variable expandedGateWeights = expandGateWeights(gateWeights, moeOutput.getValue().getShape());
        
        // 计算1 - gateWeight
        Variable oneMinusGate = createOnesLike(expandedGateWeights).sub(expandedGateWeights);
        
        // 门控计算
        Variable gatedMoE = expandedGateWeights.mul(moeOutput);
        Variable gatedOriginal = oneMinusGate.mul(original);
        
        return gatedMoE.add(gatedOriginal);
    }
    
    /**
     * 扩展门控权重维度
     */
    private Variable expandGateWeights(Variable gateWeights, Shape targetShape) {
        NdArray gateData = gateWeights.getValue();
        NdArray targetData = NdArray.zeros(targetShape);
        
        double[] gateArray = gateData.toDoubleArray();
        double[] targetArray = targetData.toDoubleArray();
        
        int batchSize = targetShape.getDimension(0);
        int seqLen = targetShape.getDimension(1);
        int dModel = targetShape.getDimension(2);
        
        // 将门控权重广播到所有维度
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                double gateValue = gateArray[b * seqLen + s]; // 假设门控权重形状为[batch, seq, 1]
                for (int d = 0; d < dModel; d++) {
                    int targetIdx = (b * seqLen + s) * dModel + d;
                    targetArray[targetIdx] = gateValue;
                }
            }
        }
        
        return new Variable(targetData);
    }
    
    /**
     * 创建与给定变量形状相同的全1张量
     */
    private Variable createOnesLike(Variable x) {
        NdArray data = x.getValue();
        NdArray ones = NdArray.ones(data.getShape());
        return new Variable(ones);
    }
    
    /**
     * 计算负载均衡损失
     * 
     * @return 负载均衡损失值
     */
    public double computeLoadBalancingLoss() {
        return moeFFN.computeLoadBalanceLoss();
    }
    
    /**
     * 重置MoE统计信息
     */
    public void resetMoEStats() {
        moeFFN.resetStats();
    }
    
    /**
     * 获取专家使用统计
     * 
     * @return 专家使用计数数组
     */
    public long[] getExpertUsageCount() {
        return moeFFN.getExpertUsageCount();
    }
    
    /**
     * 获取总处理token数
     * 
     * @return 总token数
     */
    public long getTotalTokens() {
        return moeFFN.getTotalTokensProcessed();
    }
    
    /**
     * 计算总参数数量
     * 
     * @return 总参数数
     */
    public long getTotalParameterCount() {
        long totalParams = 0;
        
        // 注意力层参数
        totalParams += 4L * dModel * dModel; // Q, K, V, O投影
        
        // 层归一化参数
        totalParams += 2L * dModel * 2; // 两个LayerNorm
        
        // MoE参数（所有专家）
        long expertParams = 2L * dModel * (dModel * 4); // 假设专家维度是dModel*4
        totalParams += numExperts * expertParams;
        
        // 门控层参数
        totalParams += dModel;
        
        return totalParams;
    }
    
    /**
     * 计算激活参数数量
     * 
     * @return 激活参数数
     */
    public long getActiveParameterCount() {
        long activeParams = 0;
        
        // 注意力层参数（全部激活）
        activeParams += 4L * dModel * dModel;
        
        // 层归一化参数（全部激活）
        activeParams += 2L * dModel * 2;
        
        // MoE参数（只有部分专家激活，假设Top-2）
        long expertParams = 2L * dModel * (dModel * 4);
        activeParams += 2 * expertParams; // Top-2
        
        // 门控层参数（全部激活）
        activeParams += dModel;
        
        return activeParams;
    }
    
    /**
     * 生成层状态报告
     * 
     * @return 状态报告字符串
     */
    public String getLayerReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== V3 Transformer Block 报告 ===\n");
        sb.append(String.format("层名称: %s\n", name));
        sb.append(String.format("模型维度: %d, 注意力头数: %d\n", dModel, numHeads));
        sb.append(String.format("专家数量: %d, 当前任务: %s\n", 
            numExperts, currentTaskType != null ? currentTaskType.getValue() : "未知"));
        sb.append(String.format("总参数: %d, 激活参数: %d\n", 
            getTotalParameterCount(), getActiveParameterCount()));
        sb.append(String.format("处理Token数: %d\n", getTotalTokens()));
        sb.append(String.format("负载均衡损失: %.6f\n", computeLoadBalancingLoss()));
        
        if (lastRoutingInfo != null) {
            sb.append("最近路由信息:\n");
            sb.append(lastRoutingInfo.getSummary());
        }
        
        sb.append("==============================");
        return sb.toString();
    }
    
    @Override
    public void clearGrads() {
        attention.clearGrads();
        norm1.clearGrads();
        norm2.clearGrads();
        moeFFN.clearGrads();
        gateLayer.clearGrads();
    }
    
    // ========== Getter Methods ==========
    
    public MultiHeadAttention getAttention() { return attention; }
    public MixtureOfExperts getMoeFFN() { return moeFFN; }
    public ExpertRoutingInfo getLastRoutingInfo() { return lastRoutingInfo; }
    public TaskType getCurrentTaskType() { return currentTaskType; }
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getNumExperts() { return numExperts; }
}