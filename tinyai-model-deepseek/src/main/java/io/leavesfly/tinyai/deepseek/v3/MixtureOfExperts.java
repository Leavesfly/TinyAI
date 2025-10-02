package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.activate.ReLuLayer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 混合专家模型(MoE)层
 * 
 * 实现了DeepSeek V3的混合专家架构，包含：
 * 1. 路由网络 - 根据输入选择合适的专家
 * 2. 多个专家网络 - 不同专家擅长不同任务
 * 3. 负载均衡机制 - 确保专家使用的均衡性
 * 4. 任务类型感知 - 根据任务类型调整专家选择偏置
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MixtureOfExperts extends Block {
    
    /**
     * 模型维度
     */
    private final int dModel;
    
    /**
     * 专家数量
     */
    private final int numExperts;
    
    /**
     * 选择的专家数量（top-k）
     */
    private final int numSelected;
    
    /**
     * 专家容量因子
     */
    private final float expertCapacityFactor;
    
    /**
     * 路由网络 - 将输入映射到专家权重
     */
    private LinearLayer router;
    
    /**
     * 专家网络列表
     */
    private List<ExpertNetwork> experts;
    
    /**
     * 专家特化类型映射
     */
    private Map<Integer, TaskType> expertSpecializations;
    
    /**
     * 构造函数
     * 
     * @param name 块名称
     * @param dModel 模型维度
     * @param numExperts 专家数量
     * @param numSelected 选择的专家数量
     * @param expertCapacityFactor 专家容量因子
     */
    public MixtureOfExperts(String name, int dModel, int numExperts, int numSelected, 
                           float expertCapacityFactor) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        this.dModel = dModel;
        this.numExperts = numExperts;
        this.numSelected = numSelected;
        this.expertCapacityFactor = expertCapacityFactor;
        
        init();
    }
    
    /**
     * 默认构造函数 - 使用标准配置
     */
    public MixtureOfExperts(String name, int dModel) {
        this(name, dModel, 8, 2, 1.0f);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 路由网络：将输入映射到专家权重
            router = new LinearLayer(name + "_router", dModel, numExperts, false);
            addLayer(router);
            
            // 创建专家网络
            experts = new ArrayList<>();
            for (int i = 0; i < numExperts; i++) {
                ExpertNetwork expert = new ExpertNetwork(name + "_expert_" + i, i, dModel, dModel * 4);
                experts.add(expert);
                addLayer(expert);
            }
            
            // 初始化专家特化类型
            initializeExpertSpecializations();
            
            alreadyInit = true;
        }
    }
    
    /**
     * 初始化专家特化类型映射
     */
    private void initializeExpertSpecializations() {
        expertSpecializations = new HashMap<>();
        TaskType[] taskTypes = TaskType.values();
        
        for (int i = 0; i < numExperts; i++) {
            // 将专家分配给不同任务类型，如果专家数量多于任务类型，则重复分配
            TaskType taskType = taskTypes[i % taskTypes.length];
            expertSpecializations.put(i, taskType);
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        TaskType taskType = null;
        
        // 如果有第二个参数，认为是任务类型标识（简化处理）
        if (inputs.length > 1) {
            // 这里可以扩展任务类型传递机制
            taskType = TaskType.GENERAL; // 默认使用通用任务
        }
        
        // 执行MoE前向传播
        MoEResult result = forwardWithTaskType(input, taskType);
        
        return result.output;
    }
    
    /**
     * 执行带任务类型感知的MoE前向传播
     * 
     * @param input 输入变量
     * @param taskType 任务类型（可为null）
     * @return MoE结果，包含输出和路由信息
     */
    public MoEResult forwardWithTaskType(Variable input, TaskType taskType) {
        NdArray inputData = input.getValue();
        int batchSize = inputData.getShape().getDimension(0);
        int seqLen = inputData.getShape().getDimension(1);
        
        // 重塑为2D进行路由计算
        NdArray inputFlat = inputData.reshape(Shape.of(batchSize * seqLen, dModel));
        Variable flatInput = new Variable(inputFlat);
        
        // 路由计算
        Variable routerLogits = router.layerForward(flatInput);
        NdArray routerLogitsData = routerLogits.getValue();
        
        // 任务类型偏置
        if (taskType != null) {
            applyTaskTypeBias(routerLogitsData, taskType);
        }
        
        // 计算专家权重和选择
        NdArray routerProbs = routerLogitsData.softMax();
        TopKResult topK = computeTopK(routerProbs, numSelected);
        
        // 执行专家计算
        NdArray output = computeExpertOutputs(inputFlat, topK);
        
        // 重塑回原始形状
        output = output.reshape(Shape.of(batchSize, seqLen, dModel));
        
        // 计算负载均衡损失
        float loadBalanceLoss = computeLoadBalanceLoss(routerProbs);
        
        // 创建路由信息
        ExpertRoutingInfo routingInfo = new ExpertRoutingInfo(
            topK.weights, topK.indices, 0.0f, loadBalanceLoss);
        
        return new MoEResult(new Variable(output), routingInfo);
    }
    
    /**
     * 应用任务类型偏置
     */
    private void applyTaskTypeBias(NdArray routerLogits, TaskType taskType) {
        float biasValue = 0.5f; // 偏置强度
        
        for (int i = 0; i < numExperts; i++) {
            TaskType expertType = expertSpecializations.get(i);
            if (expertType == taskType) {
                // 为相关专家添加正偏置
                for (int j = 0; j < routerLogits.getShape().getDimension(0); j++) {
                    float currentValue = routerLogits.get(j, i);
                    routerLogits.set(currentValue + biasValue, j, i);
                }
            }
        }
    }
    
    /**
     * 计算Top-K专家选择
     */
    private TopKResult computeTopK(NdArray probs, int k) {
        int batchSeqLen = probs.getShape().getDimension(0);
        NdArray weights = NdArray.of(Shape.of(batchSeqLen, k));
        List<Integer> allIndices = new ArrayList<>();
        
        for (int i = 0; i < batchSeqLen; i++) {
            // 为每个样本找到top-k专家
            float[] expertProbs = new float[numExperts];
            for (int j = 0; j < numExperts; j++) {
                expertProbs[j] = probs.get(i, j);
            }
            
            // 简单的top-k选择（可以优化）
            int[] topIndices = findTopK(expertProbs, k);
            float totalWeight = 0.0f;
            
            // 计算选中专家的总权重
            for (int idx : topIndices) {
                totalWeight += expertProbs[idx];
            }
            
            // 归一化权重并存储
            for (int j = 0; j < k; j++) {
                if (j < topIndices.length) {
                    float normalizedWeight = totalWeight > 0 ? expertProbs[topIndices[j]] / totalWeight : 0;
                    weights.set(normalizedWeight, i, j);
                    allIndices.add(topIndices[j]);
                } else {
                    weights.set(0.0f, i, j);
                    allIndices.add(-1);
                }
            }
        }
        
        return new TopKResult(weights, allIndices);
    }
    
    /**
     * 找到数组中的top-k最大值索引
     */
    private int[] findTopK(float[] values, int k) {
        List<IndexValue> indexValues = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            indexValues.add(new IndexValue(i, values[i]));
        }
        
        // 按值降序排序
        indexValues.sort((a, b) -> Float.compare(b.value, a.value));
        
        int[] result = new int[Math.min(k, values.length)];
        for (int i = 0; i < result.length; i++) {
            result[i] = indexValues.get(i).index;
        }
        
        return result;
    }
    
    /**
     * 计算专家输出
     */
    private NdArray computeExpertOutputs(NdArray input, TopKResult topK) {
        int batchSeqLen = input.getShape().getDimension(0);
        NdArray output = NdArray.zeros(Shape.of(batchSeqLen, dModel));
        
        for (int i = 0; i < batchSeqLen; i++) {
            for (int j = 0; j < numSelected; j++) {
                int expertIdx = topK.indices.get(i * numSelected + j);
                if (expertIdx >= 0 && expertIdx < numExperts) {
                    float weight = topK.weights.get(i, j);
                    
                    // 获取当前样本的输入（手动提取行数据）
                    NdArray sampleInput = NdArray.of(Shape.of(1, dModel));
                    for (int d = 0; d < dModel; d++) {
                        sampleInput.set(input.get(i, d), 0, d);
                    }
                    Variable sampleVar = new Variable(sampleInput);
                    
                    // 通过专家网络前向传播
                    Variable expertOutput = experts.get(expertIdx).layerForward(sampleVar);
                    NdArray expertResult = expertOutput.getValue();
                    
                    // 加权累加到输出
                    for (int d = 0; d < dModel; d++) {
                        float currentValue = output.get(i, d);
                        float expertValue = expertResult.get(0, d);
                        output.set(currentValue + weight * expertValue, i, d);
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * 计算负载均衡损失
     */
    private float computeLoadBalanceLoss(NdArray routerProbs) {
        // 计算每个专家的平均使用率
        NdArray expertUsage = NdArray.zeros(Shape.of(numExperts));
        int totalSamples = routerProbs.getShape().getDimension(0);
        
        for (int i = 0; i < numExperts; i++) {
            float usage = 0.0f;
            for (int j = 0; j < totalSamples; j++) {
                usage += routerProbs.get(j, i);
            }
            expertUsage.set(usage / totalSamples, i);
        }
        
        // 计算与均匀分布的KL散度
        float targetUsage = 1.0f / numExperts;
        float klLoss = 0.0f;
        
        for (int i = 0; i < numExperts; i++) {
            float usage = expertUsage.get(i) + 1e-8f; // 避免log(0)
            klLoss += usage * (float)Math.log(usage / targetUsage);
        }
        
        return klLoss;
    }
    
    // 内部辅助类
    private static class TopKResult {
        final NdArray weights;
        final List<Integer> indices;
        
        TopKResult(NdArray weights, List<Integer> indices) {
            this.weights = weights;
            this.indices = indices;
        }
    }
    
    private static class IndexValue {
        final int index;
        final float value;
        
        IndexValue(int index, float value) {
            this.index = index;
            this.value = value;
        }
    }
    
    /**
     * MoE结果包装类
     */
    public static class MoEResult {
        public final Variable output;
        public final ExpertRoutingInfo routingInfo;
        
        public MoEResult(Variable output, ExpertRoutingInfo routingInfo) {
            this.output = output;
            this.routingInfo = routingInfo;
        }
    }
    
    /**
     * 专家网络实现
     */
    private static class ExpertNetwork extends Block {
        private final int expertId;
        private final int dModel;
        private final int dExpert;
        
        private LinearLayer firstLinear;
        private ReLuLayer activation;
        private LinearLayer secondLinear;
        
        public ExpertNetwork(String name, int expertId, int dModel, int dExpert) {
            super(name, Shape.of(-1, dModel), Shape.of(-1, dModel));
            this.expertId = expertId;
            this.dModel = dModel;
            this.dExpert = dExpert;
            init();
        }
        
        @Override
        public void init() {
            if (!alreadyInit) {
                // 第一个线性层：dModel -> dExpert
                firstLinear = new LinearLayer(name + "_linear1", dModel, dExpert, true);
                addLayer(firstLinear);
                
                // ReLU激活函数
                activation = new ReLuLayer(name + "_relu", Shape.of(-1, dExpert));
                addLayer(activation);
                
                // 第二个线性层：dExpert -> dModel
                secondLinear = new LinearLayer(name + "_linear2", dExpert, dModel, true);
                addLayer(secondLinear);
                
                alreadyInit = true;
            }
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable input = inputs[0];
            
            // 第一个线性变换
            Variable hidden = firstLinear.layerForward(input);
            
            // ReLU激活
            Variable activated = activation.layerForward(hidden);
            
            // 第二个线性变换
            return secondLinear.layerForward(activated);
        }
    }
    
    // Getters
    public int getDModel() {
        return dModel;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public int getNumSelected() {
        return numSelected;
    }
    
    public Map<Integer, TaskType> getExpertSpecializations() {
        return expertSpecializations;
    }
}