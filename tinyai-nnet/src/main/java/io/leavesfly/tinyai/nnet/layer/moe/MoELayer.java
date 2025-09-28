package io.leavesfly.tinyai.nnet.layer.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE混合专家层
 * <p>
 * 这是Mixture of Experts (MoE)架构的核心层，整合了多个专家网络和门控网络。
 * 它负责协调专家的选择、激活和输出的聚合。
 * <p>
 * 工作流程：
 * 1. 门控网络为每个token选择Top-K专家
 * 2. 只有被选中的专家才会被激活
 * 3. 将专家输出按门控权重进行加权求和
 * 4. 返回最终的混合输出
 * <p>
 * 特点：
 * - 稀疏激活：只计算选中的专家
 * - 动态路由：每个token可能使用不同的专家组合
 * - 负载均衡：通过门控网络分配负载
 * - 可扩展性：通过增加专家数量扩展模型容量
 *
 * @author leavesfly
 * @version 1.0
 */
public class MoELayer extends Layer {
    
    private List<Expert> experts;           // 专家网络列表
    private GateNetwork gateNetwork;        // 门控网络
    
    private int numExperts;                 // 专家数量
    private int dModel;                     // 输入/输出维度
    private int dExpert;                    // 专家隐藏层维度
    private int topK;                       // Top-K专家选择
    
    // 统计信息
    private long totalTokens;               // 处理的总token数
    private long[] expertUsageCount;        // 每个专家的使用次数
    
    /**
     * 构造MoE层
     * 
     * @param name 层名称
     * @param numExperts 专家数量
     * @param dModel 输入/输出维度
     * @param dExpert 专家隐藏层维度
     * @param topK Top-K专家选择数量
     * @param useNoise 门控是否使用噪声
     * @param noiseEpsilon 噪声强度
     */
    public MoELayer(String name, int numExperts, int dModel, int dExpert, int topK,
                   boolean useNoise, double noiseEpsilon) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (numExperts <= 0) {
            throw new IllegalArgumentException("专家数量必须大于0");
        }
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException("topK必须在1到numExperts之间");
        }
        
        this.numExperts = numExperts;
        this.dModel = dModel;
        this.dExpert = dExpert;
        this.topK = topK;
        this.expertUsageCount = new long[numExperts];
        
        init();
        initializeExperts(useNoise, noiseEpsilon);
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public MoELayer(String name, int numExperts, int dModel, int dExpert) {
        this(name, numExperts, dModel, dExpert, 2, true, 0.1);
    }
    
    /**
     * 简化的构造函数（默认专家隐藏层维度为4倍dModel）
     */
    public MoELayer(String name, int numExperts, int dModel) {
        this(name, numExperts, dModel, dModel * 4);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            alreadyInit = true;
        }
    }
    
    /**
     * 初始化专家网络和门控网络
     */
    private void initializeExperts(boolean useNoise, double noiseEpsilon) {
        // 初始化门控网络
        gateNetwork = new GateNetwork(
            name + "_gate", 
            dModel, 
            numExperts, 
            topK, 
            useNoise, 
            noiseEpsilon, 
            true  // 启用负载均衡
        );
        
        // 初始化专家网络列表
        experts = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            Expert expert = new Expert(
                name + "_expert", 
                i,  // 专家ID
                dModel, 
                dExpert
            );
            experts.add(expert);
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputData = input.getValue();
        
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 验证输入维度
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException(
                String.format("MoE层输入维度不匹配。期望%d，实际%d", 
                             dModel, inputData.getShape().getDimension(2))
            );
        }
        
        // 1. 使用门控网络选择专家
        GateNetwork.GateOutput gateOutput = gateNetwork.selectTopKExperts(input);
        
        // 2. 为每个token计算专家输出
        NdArray finalOutput = NdArray.zeros(inputData.getShape());
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 获取当前token的输入
                NdArray tokenInput = extractTokenInput(inputData, b, s);
                Variable tokenVar = new Variable(tokenInput);
                
                // 计算选中专家的加权输出
                NdArray tokenOutput = computeTokenOutput(
                    tokenVar, 
                    gateOutput.expertIndices[b][s], 
                    gateOutput.expertWeights[b][s]
                );
                
                // 将结果写回最终输出
                setTokenOutput(finalOutput, tokenOutput, b, s);
                
                // 更新专家使用统计
                updateExpertUsageStats(gateOutput.expertIndices[b][s]);
            }
        }
        
        totalTokens += batchSize * seqLen;
        return new Variable(finalOutput);
    }
    
    /**
     * 提取单个token的输入
     */
    private NdArray extractTokenInput(NdArray batchInput, int batchIdx, int seqIdx) {
        // 从(batch_size, seq_len, dModel)中提取(1, 1, dModel)
        NdArray tokenInput = NdArray.of(Shape.of(1, 1, dModel));
        for (int d = 0; d < dModel; d++) {
            float value = batchInput.get(batchIdx, seqIdx, d);
            tokenInput.set(value, 0, 0, d);
        }
        return tokenInput;
    }
    
    /**
     * 计算单个token通过选中专家的输出
     */
    private NdArray computeTokenOutput(Variable tokenInput, int[] expertIndices, float[] expertWeights) {
        NdArray weightedSum = NdArray.zeros(Shape.of(1, 1, dModel));
        
        // 对每个选中的专家计算输出并加权求和
        for (int k = 0; k < topK; k++) {
            int expertIdx = expertIndices[k];
            float weight = expertWeights[k];
            
            if (weight > 0.0f && expertIdx >= 0 && expertIdx < numExperts) {
                // 通过专家计算输出
                Expert expert = experts.get(expertIdx);
                Variable expertOutput = expert.layerForward(tokenInput);
                NdArray expertOutputData = expertOutput.getValue();
                
                // 加权累加到最终输出
                for (int d = 0; d < dModel; d++) {
                    float currentValue = weightedSum.get(0, 0, d);
                    float expertValue = expertOutputData.get(0, 0, d);
                    weightedSum.set(currentValue + weight * expertValue, 0, 0, d);
                }
            }
        }
        
        return weightedSum;
    }
    
    /**
     * 将token输出设置到最终输出中
     */
    private void setTokenOutput(NdArray finalOutput, NdArray tokenOutput, int batchIdx, int seqIdx) {
        for (int d = 0; d < dModel; d++) {
            float value = tokenOutput.get(0, 0, d);
            finalOutput.set(value, batchIdx, seqIdx, d);
        }
    }
    
    /**
     * 更新专家使用统计
     */
    private void updateExpertUsageStats(int[] expertIndices) {
        for (int expertIdx : expertIndices) {
            if (expertIdx >= 0 && expertIdx < numExperts) {
                expertUsageCount[expertIdx]++;
            }
        }
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // MoE层的反向传播需要根据门控权重分配梯度到相应的专家
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    /**
     * 获取专家负载均衡统计信息
     */
    public LoadBalancingStats getLoadBalancingStats() {
        if (totalTokens == 0) {
            return new LoadBalancingStats(expertUsageCount, 0, 0.0, 0.0);
        }
        
        // 计算平均使用率
        double averageUsage = (double) totalTokens / numExperts;
        
        // 计算负载不均衡系数（标准差）
        double variance = 0.0;
        for (long usage : expertUsageCount) {
            double diff = usage - averageUsage;
            variance += diff * diff;
        }
        double standardDeviation = Math.sqrt(variance / numExperts);
        double loadImbalance = standardDeviation / averageUsage;
        
        return new LoadBalancingStats(expertUsageCount, totalTokens, averageUsage, loadImbalance);
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        totalTokens = 0;
        for (int i = 0; i < numExperts; i++) {
            expertUsageCount[i] = 0;
        }
    }
    
    // Getter方法
    public int getNumExperts() { return numExperts; }
    public int getDModel() { return dModel; }
    public int getDExpert() { return dExpert; }
    public int getTopK() { return topK; }
    public List<Expert> getExperts() { return experts; }
    public GateNetwork getGateNetwork() { return gateNetwork; }
    public long getTotalTokens() { return totalTokens; }
    public long[] getExpertUsageCount() { return expertUsageCount.clone(); }
    
    /**
     * 负载均衡统计信息类
     */
    public static class LoadBalancingStats {
        public final long[] expertUsageCount;
        public final long totalTokens;
        public final double averageUsage;
        public final double loadImbalance;  // 0.0表示完全均衡，值越大表示越不均衡
        
        public LoadBalancingStats(long[] expertUsageCount, long totalTokens, 
                                double averageUsage, double loadImbalance) {
            this.expertUsageCount = expertUsageCount.clone();
            this.totalTokens = totalTokens;
            this.averageUsage = averageUsage;
            this.loadImbalance = loadImbalance;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("LoadBalancingStats{\n");
            sb.append("  totalTokens=").append(totalTokens).append("\n");
            sb.append("  averageUsage=").append(String.format("%.2f", averageUsage)).append("\n");
            sb.append("  loadImbalance=").append(String.format("%.4f", loadImbalance)).append("\n");
            sb.append("  expertUsage=[");
            for (int i = 0; i < expertUsageCount.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(expertUsageCount[i]);
            }
            sb.append("]\n}");
            return sb.toString();
        }
    }
    
    @Override
    public String toString() {
        return String.format("MoELayer(numExperts=%d, dModel=%d, dExpert=%d, topK=%d, totalTokens=%d)", 
                           numExperts, dModel, dExpert, topK, totalTokens);
    }
}