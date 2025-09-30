package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.math.Exp;
import io.leavesfly.tinyai.func.math.Log;
import io.leavesfly.tinyai.func.matrix.MatMul;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.LayerAble;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;
import io.leavesfly.tinyai.func.math.ReLu;
import io.leavesfly.tinyai.func.math.GELU;

import java.util.*;

/**
 * 混合专家模型(MoE)层实现
 * 
 * 基于DeepSeek V3架构的专家混合机制，支持：
 * 1. Top-K专家选择
 * 2. 任务类型感知的专家路由
 * 3. 负载均衡机制
 * 4. 专家特化(不同专家擅长不同任务)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MixtureOfExperts extends LayerAble {
    
    // ========== 配置参数 ==========
    private int dModel;                    // 模型维度
    private int numExperts;                // 专家总数
    private int numSelected;               // 选择的专家数量(Top-K)
    private double expertCapacityFactor;   // 专家容量因子
    private double loadBalanceWeight;      // 负载均衡权重
    
    // ========== 网络组件 ==========
    private LinearLayer router;                 // 路由网络
    private List<LayerAble> experts;       // 专家网络列表
    
    // ========== 专家特化配置 ==========
    private Map<Integer, TaskType> expertSpecializations; // 专家特化映射
    
    // ========== 运行时统计 ==========
    private long[] expertUsageCount;      // 专家使用计数
    private double[] expertTotalWeights;  // 专家总权重统计
    private long totalTokensProcessed;    // 处理的总token数
    
    /**
     * 构造函数
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numExperts 专家数量
     * @param numSelected 选择的专家数量
     * @param expertCapacityFactor 专家容量因子
     */
    public MixtureOfExperts(String name, int dModel, int numExperts, 
                           int numSelected, double expertCapacityFactor) {
        this.name = name;
        this.dModel = dModel;
        this.numExperts = numExperts;
        this.numSelected = numSelected;
        this.expertCapacityFactor = expertCapacityFactor;
        this.loadBalanceWeight = 0.01; // 默认负载均衡权重
        
        // 初始化统计数据
        this.expertUsageCount = new long[numExperts];
        this.expertTotalWeights = new double[numExperts];
        this.totalTokensProcessed = 0L;
        
        // 设置输入输出形状
        this.inputShape = Shape.of(-1, -1, dModel);  // [batch, seq, dModel]
        this.outputShape = Shape.of(-1, -1, dModel); // [batch, seq, dModel]
        
        init();
    }
    
    /**
     * 默认构造函数
     * 
     * @param name 层名称
     * @param dModel 模型维度
     * @param numExperts 专家数量
     */
    public MixtureOfExperts(String name, int dModel, int numExperts) {
        this(name, dModel, numExperts, 2, 1.0); // 默认Top-2
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 1. 初始化路由网络
            router = new LinearLayer(name + "_router", dModel, numExperts, false);
            router.init();
            
            // 2. 初始化专家网络
            experts = new ArrayList<>(numExperts);
            for (int i = 0; i < numExperts; i++) {
                LayerAble expert = createExpert(name + "_expert_" + i, dModel);
                expert.init();
                experts.add(expert);
            }
            
            // 3. 初始化专家特化配置
            initializeExpertSpecializations();
            
            alreadyInit = true;
        }
    }
    
    /**
     * 创建单个专家网络
     * 
     * @param expertName 专家名称
     * @param dModel 模型维度
     * @return 专家网络
     */
    private LayerAble createExpert(String expertName, int dModel) {
        // 创建一个简单的前馈网络作为专家
        // 实际实现中可以根据需要创建更复杂的专家结构
        int dExpert = (int) (dModel * 2.7); // DeepSeek惯例：专家维度是模型维度的2.7倍
        
        return new ExpertNetwork(expertName, dModel, dExpert);
    }
    
    /**
     * 初始化专家特化配置
     * 为不同专家分配不同的任务类型特长
     */
    private void initializeExpertSpecializations() {
        expertSpecializations = new HashMap<>();
        
        // 根据专家数量分配特化
        TaskType[] taskTypes = TaskType.values();
        for (int i = 0; i < numExperts; i++) {
            TaskType taskType = taskTypes[i % taskTypes.length];
            expertSpecializations.put(i, taskType);
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];  // [batch_size, seq_len, d_model]
        NdArray inputData = input.getValue();
        Shape inputShape = inputData.getShape();
        
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int dModel = inputShape.getDimension(2);
        
        // 更新token处理统计
        totalTokensProcessed += (long) batchSize * seqLen;
        
        // 1. 重塑为2D进行路由 [batch*seq, d_model]
        Variable flatInput = input.reshape(Shape.of(batchSize * seqLen, dModel));
        
        // 2. 路由计算 [batch*seq, num_experts]
        Variable routerLogits = router.layerForward(flatInput);
        
        // 3. 任务类型偏置（如果提供了任务类型）
        TaskType taskType = extractTaskTypeFromInputs(inputs);
        if (taskType != null) {
            routerLogits = applyTaskTypeBias(routerLogits, taskType);
        }
        
        // 4. Softmax归一化得到专家概率
        Variable routerProbs = routerLogits.softMax();
        
        // 5. Top-K专家选择
        TopKResult topKResult = selectTopKExperts(routerProbs, numSelected);
        
        // 6. 专家计算
        Variable output = computeExpertOutputs(flatInput, topKResult);
        
        // 7. 重塑回原始形状
        output = output.reshape(Shape.of(batchSize, seqLen, dModel));
        
        // 8. 更新专家使用统计
        updateExpertStatistics(topKResult);
        
        return output;
    }
    
    /**
     * 从输入中提取任务类型信息
     * 这里简化实现，实际可以通过更复杂的方式推断任务类型
     */
    private TaskType extractTaskTypeFromInputs(Variable[] inputs) {
        // 简化实现：如果有第二个输入参数包含任务类型信息
        if (inputs.length > 1 && inputs[1] != null) {
            // 这里可以实现更复杂的任务类型推断逻辑
            return TaskType.GENERAL; // 默认返回通用任务
        }
        return null;
    }
    
    /**
     * 应用任务类型偏置
     * 为与当前任务相关的专家增加权重
     */
    private Variable applyTaskTypeBias(Variable routerLogits, TaskType taskType) {
        NdArray biasArray = NdArray.zeros(routerLogits.getValue().getShape());
        double[] biasData = biasArray.toDoubleArray();
        
        // 为特化于当前任务类型的专家加权
        for (Map.Entry<Integer, TaskType> entry : expertSpecializations.entrySet()) {
            int expertIdx = entry.getKey();
            TaskType expertType = entry.getValue();
            
            if (expertType == taskType) {
                // 为匹配的专家增加0.5的偏置
                for (int i = 0; i < biasData.length; i += numExperts) {
                    if (i + expertIdx < biasData.length) {
                        biasData[i + expertIdx] += 0.5;
                    }
                }
            }
        }
        
        Variable bias = new Variable(biasArray);
        return routerLogits.add(bias);
    }
    
    /**
     * Top-K专家选择
     */
    private TopKResult selectTopKExperts(Variable routerProbs, int k) {
        NdArray probsData = routerProbs.getValue();
        int batchSeqLen = probsData.getShape().getDimension(0);
        
        // 简化实现：选择概率最高的k个专家
        List<List<Integer>> selectedExperts = new ArrayList<>();
        List<List<Double>> selectedWeights = new ArrayList<>();
        
        double[] probsArray = probsData.toDoubleArray();
        
        for (int i = 0; i < batchSeqLen; i++) {
            // 获取当前token的专家概率
            double[] tokenProbs = new double[numExperts];
            System.arraycopy(probsArray, i * numExperts, tokenProbs, 0, numExperts);
            
            // 找到Top-K专家
            List<Integer> topExperts = new ArrayList<>();
            List<Double> topWeights = new ArrayList<>();
            
            // 简单的Top-K选择
            for (int j = 0; j < k && j < numExperts; j++) {
                int maxIdx = 0;
                double maxProb = tokenProbs[0];
                for (int expertIdx = 1; expertIdx < numExperts; expertIdx++) {
                    if (tokenProbs[expertIdx] > maxProb) {
                        maxProb = tokenProbs[expertIdx];
                        maxIdx = expertIdx;
                    }
                }
                
                topExperts.add(maxIdx);
                topWeights.add(maxProb);
                tokenProbs[maxIdx] = -1.0; // 标记已选择
            }
            
            // 重新归一化权重
            double weightSum = topWeights.stream().mapToDouble(Double::doubleValue).sum();
            if (weightSum > 0) {
                topWeights.replaceAll(weight -> weight / weightSum);
            }
            
            selectedExperts.add(topExperts);
            selectedWeights.add(topWeights);
        }
        
        return new TopKResult(selectedExperts, selectedWeights);
    }
    
    /**
     * 计算专家输出
     */
    private Variable computeExpertOutputs(Variable flatInput, TopKResult topKResult) {
        NdArray inputData = flatInput.getValue();
        int batchSeqLen = inputData.getShape().getDimension(0);
        
        // 初始化输出
        NdArray outputData = NdArray.zeros(inputData.getShape());
        double[] outputArray = outputData.toDoubleArray();
        double[] inputArray = inputData.toDoubleArray();
        
        // 为每个token计算专家输出
        for (int tokenIdx = 0; tokenIdx < batchSeqLen; tokenIdx++) {
            List<Integer> experts = topKResult.selectedExperts.get(tokenIdx);
            List<Double> weights = topKResult.selectedWeights.get(tokenIdx);
            
            // 提取当前token的输入
            double[] tokenInput = new double[dModel];
            System.arraycopy(inputArray, tokenIdx * dModel, tokenInput, 0, dModel);
            Variable tokenInputVar = new Variable(NdArray.of(tokenInput).reshape(Shape.of(1, dModel)));
            
            // 计算各专家的加权输出
            double[] tokenOutput = new double[dModel];
            
            for (int i = 0; i < experts.size(); i++) {
                int expertIdx = experts.get(i);
                double weight = weights.get(i);
                
                // 计算专家输出
                Variable expertOutput = this.experts.get(expertIdx).layerForward(tokenInputVar);
                double[] expertOutputArray = expertOutput.getValue().toDoubleArray();
                
                // 加权累加
                for (int j = 0; j < dModel; j++) {
                    tokenOutput[j] += weight * expertOutputArray[j];
                }
            }
            
            // 写入最终输出
            System.arraycopy(tokenOutput, 0, outputArray, tokenIdx * dModel, dModel);
        }
        
        return new Variable(outputData);
    }
    
    /**
     * 更新专家使用统计
     */
    private void updateExpertStatistics(TopKResult topKResult) {
        for (List<Integer> experts : topKResult.selectedExperts) {
            for (int expertIdx : experts) {
                expertUsageCount[expertIdx]++;
            }
        }
        
        for (int i = 0; i < topKResult.selectedWeights.size(); i++) {
            List<Integer> experts = topKResult.selectedExperts.get(i);
            List<Double> weights = topKResult.selectedWeights.get(i);
            
            for (int j = 0; j < experts.size(); j++) {
                int expertIdx = experts.get(j);
                double weight = weights.get(j);
                expertTotalWeights[expertIdx] += weight;
            }
        }
    }
    
    /**
     * 计算负载均衡损失
     */
    public double computeLoadBalanceLoss() {
        if (totalTokensProcessed == 0) {
            return 0.0;
        }
        
        // 计算每个专家的平均使用率
        double[] expertUsageRates = new double[numExperts];
        for (int i = 0; i < numExperts; i++) {
            expertUsageRates[i] = (double) expertUsageCount[i] / totalTokensProcessed;
        }
        
        // 理想情况下每个专家使用率应该相等
        double targetUsageRate = 1.0 / numExperts;
        
        // 计算KL散度作为负载均衡损失
        double klDiv = 0.0;
        for (double rate : expertUsageRates) {
            if (rate > 1e-8) {
                klDiv += rate * Math.log(rate / targetUsageRate);
            }
        }
        
        return klDiv * loadBalanceWeight;
    }
    
    /**
     * 重置MoE统计信息
     */
    public void resetStats() {
        Arrays.fill(expertUsageCount, 0L);
        Arrays.fill(expertTotalWeights, 0.0);
        totalTokensProcessed = 0L;
    }
    
    /**
     * 获取专家使用分布报告
     */
    public String getExpertUsageReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== MoE专家使用分布报告 ===\n");
        sb.append(String.format("总处理Token数: %d\n", totalTokensProcessed));
        sb.append("专家使用统计:\n");
        
        for (int i = 0; i < numExperts; i++) {
            double usageRate = totalTokensProcessed > 0 ? 
                (double) expertUsageCount[i] / totalTokensProcessed : 0.0;
            TaskType specialization = expertSpecializations.get(i);
            
            sb.append(String.format("  专家 %d [%s]: 使用次数=%d, 使用率=%.3f, 总权重=%.3f\n",
                i, specialization != null ? specialization.getValue() : "unknown",
                expertUsageCount[i], usageRate, expertTotalWeights[i]));
        }
        
        sb.append(String.format("负载均衡损失: %.6f\n", computeLoadBalanceLoss()));
        sb.append("========================");
        
        return sb.toString();
    }
    
    @Override
    public void clearGrads() {
        router.clearGrads();
        for (LayerAble expert : experts) {
            expert.clearGrads();
        }
    }
    
    // ========== Inner Classes ==========
    
    /**
     * Top-K选择结果
     */
    private static class TopKResult {
        List<List<Integer>> selectedExperts;
        List<List<Double>> selectedWeights;
        
        TopKResult(List<List<Integer>> selectedExperts, List<List<Double>> selectedWeights) {
            this.selectedExperts = selectedExperts;
            this.selectedWeights = selectedWeights;
        }
    }
    
    /**
     * 专家网络实现
     */
    private static class ExpertNetwork extends LayerAble {
        private Linear layer1;
        private GELU activation;
        private Linear layer2;
        
        public ExpertNetwork(String name, int dModel, int dExpert) {
            this.name = name;
            this.inputShape = Shape.of(-1, dModel);
            this.outputShape = Shape.of(-1, dModel);
            init();
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                layer1 = new LinearLayer(name + "_layer1", inputShape.getDimension(-1), 
                    (int)(inputShape.getDimension(-1) * 2.7), false);
                activation = new Variable(NdArray.zeros(Shape.of(1, 1))); // 传统激活函数
                layer2 = new LinearLayer(name + "_layer2", (int)(inputShape.getDimension(-1) * 2.7), 
                    inputShape.getDimension(-1), false);
                
                layer1.init();
                activation.init();
                layer2.init();
                
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable x = inputs[0];
            x = layer1.layerForward(x);
            x = new Variable(new GELU().forward(x.getValue())); // 使用GELU激活
            x = layer2.layerForward(x);
            return x;
        }
        
        @Override
        public void clearGrads() {
            layer1.clearGrads();
            layer2.clearGrads();
        }
    }
    
    // ========== Getter Methods ==========
    
    public int getNumExperts() { return numExperts; }
    public int getNumSelected() { return numSelected; }
    public long[] getExpertUsageCount() { return expertUsageCount; }
    public double[] getExpertTotalWeights() { return expertTotalWeights; }
    public long getTotalTokensProcessed() { return totalTokensProcessed; }
    public Map<Integer, TaskType> getExpertSpecializations() { return expertSpecializations; }
}