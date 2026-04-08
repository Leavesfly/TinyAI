package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE Layer - 混合专家层
 * 
 * 整合ExpertRouter和多个ExpertNetwork,实现完整的MoE机制
 * 
 * 工作流程:
 * 1. Router计算Top-K专家和权重
 * 2. 将输入路由到选中的专家
 * 3. 专家并行处理
 * 4. 按权重加权合并输出
 * 
 * 核心公式:
 * output = Σ(w_i · Expert_i(x)) for i in Top-K
 * 
 * @author leavesfly
 * @since 2024
 */
public class MoEBlock extends Module {
    
    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private final int numExperts;
    private final int topK;
    
    private final ExpertRouter router;
    private final List<ExpertNetwork> experts;
    
    // 统计信息
    private long[] expertUsageCount;  // 每个专家被使用次数
    private long totalCalls;          // 总调用次数
    
    /**
     * 构造函数
     * 
     * @param inputDim 输入维度
     * @param hiddenDim 专家隐藏层维度
     * @param outputDim 输出维度
     * @param numExperts 专家数量
     * @param topK Top-K选择数量
     * @param noiseFactor 路由噪声因子
     */
    public MoEBlock(int inputDim, int hiddenDim, int outputDim,
                    int numExperts, int topK, float noiseFactor) {
        super("moe_layer");
        
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.numExperts = numExperts;
        this.topK = topK;
        
        // 创建Router
        this.router = new ExpertRouter(inputDim, numExperts, topK, noiseFactor);
        registerModule("router", router);
        
        // 创建Experts
        this.experts = new ArrayList<>(numExperts);
        for (int i = 0; i < numExperts; i++) {
            ExpertNetwork expert = new ExpertNetwork(i, inputDim, hiddenDim, outputDim);
            experts.add(expert);
            registerModule("expert_" + i, expert);
        }
        
        // 初始化统计信息
        this.expertUsageCount = new long[numExperts];
        this.totalCalls = 0;
    }
    
    /**
     * 前向传播(Variable版本,Module接口要求)
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardVar(inputs[0]);
    }
    
    /**
     * 前向传播(Function接口)
     * 
     * @param inputs 输入数组
     * @return 输出NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        return forwardVar(input).getValue();
    }
    
    /**
     * 前向传播(Variable版本) - 内部自动调用Router
     * 
     * @param input 输入 [batch_size, input_dim]
     * @return 输出 [batch_size, output_dim]
     */
    public Variable forwardVar(Variable input) {
        ExpertRouter.RouterOutput routerOutput = router.forwardRouter(input);
        return forwardWithRouterOutput(input, routerOutput);
    }

    /**
     * 前向传播(接收外部RouterOutput,避免重复路由)
     * 
     * @param input 输入 [batch_size, input_dim]
     * @param routerOutput 已计算好的路由结果
     * @return 输出 [batch_size, output_dim]
     */
    public Variable forwardWithRouterOutput(Variable input, ExpertRouter.RouterOutput routerOutput) {
        // 支持3D输入 [batch_size, seq_len, input_dim] -> reshape为 [batch_size * seq_len, input_dim]
        int origBatchSize = input.getShape().getDimension(0);
        int seqLen = -1;
        Variable flatInput = input;
        if (input.getShape().getDimNum() == 3) {
            seqLen = input.getShape().getDimension(1);
            flatInput = input.reshape(Shape.of(origBatchSize * seqLen, inputDim));
        }
        
        int batchSize = flatInput.getShape().getDimension(0);
        int[][] topKIndices = routerOutput.getTopKIndices();
        float[][] topKWeights = routerOutput.getTopKWeights();
        
        // 准备输出变量 (初始化为零)
        Variable output = Variable.zeros(Shape.of(batchSize, outputDim));
        
        // 按专家分组：为每个专家收集需要处理的样本索引和对应的权重
        // expertInputs[e] 存储专家 e 需要处理的样本索引列表
        // expertWeights[e] 存储专家 e 对应样本的权重列表
        List<List<Integer>> expertBatchIndices = new ArrayList<>(numExperts);
        List<List<Float>> expertBatchWeights = new ArrayList<>(numExperts);
        
        for (int e = 0; e < numExperts; e++) {
            expertBatchIndices.add(new ArrayList<>());
            expertBatchWeights.add(new ArrayList<>());
        }
        
        // 遍历所有样本，按路由结果分组
        for (int b = 0; b < batchSize; b++) {
            for (int k = 0; k < topK; k++) {
                int expertIdx = topKIndices[b][k];
                float weight = topKWeights[b][k];
                
                expertBatchIndices.get(expertIdx).add(b);
                expertBatchWeights.get(expertIdx).add(weight);
                
                // 更新统计
                expertUsageCount[expertIdx]++;
            }
        }
        
        // 按专家批量处理
        for (int e = 0; e < numExperts; e++) {
            List<Integer> batchIndices = expertBatchIndices.get(e);
            List<Float> batchWeights = expertBatchWeights.get(e);
            
            if (batchIndices.isEmpty()) {
                continue;  // 该专家没有被选中
            }
            
            // 批量提取该专家需要处理的样本
            int expertBatchSize = batchIndices.size();
            float[] indicesArray = new float[expertBatchSize];
            for (int i = 0; i < expertBatchSize; i++) {
                indicesArray[i] = batchIndices.get(i);
            }
            
            Variable indicesVar = new Variable(NdArray.of(indicesArray));
            indicesVar.setRequireGrad(false);
            Variable expertInput = flatInput.indexSelect(0, indicesVar);  // [expert_batch_size, input_dim]
            
            // 批量调用专家
            ExpertNetwork expert = experts.get(e);
            Variable expertOutput = expert.forwardVar(expertInput);  // [expert_batch_size, output_dim]
            
            // 加权并合并到总输出
            NdArray outputData = output.getValue();
            NdArray expertOutputData = expertOutput.getValue();
            float[] outputBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) outputData).buffer;
            float[] expertBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) expertOutputData).buffer;
            
            for (int i = 0; i < expertBatchSize; i++) {
                int sampleIdx = batchIndices.get(i);
                float weight = batchWeights.get(i);
                
                // 将专家输出加权后写入对应样本位置
                for (int d = 0; d < outputDim; d++) {
                    int expertOffset = i * outputDim + d;
                    int outputOffset = sampleIdx * outputDim + d;
                    outputBuffer[outputOffset] += expertBuffer[expertOffset] * weight;
                }
            }
        }
        
        totalCalls += batchSize;
        
        // 如果原始输入是3D的，将输出reshape回 [batch_size, seq_len, output_dim]
        if (seqLen > 0) {
            output = output.reshape(Shape.of(origBatchSize, seqLen, outputDim));
        }
        
        return output;
    }
    
    /**
     * 获取负载均衡损失(由LoadBalanceLoss调用)
     * 
     * @param routerOutput Router输出
     * @return 负载均衡相关统计
     */
    public LoadBalanceStats getLoadBalanceStats(ExpertRouter.RouterOutput routerOutput) {
        float[][] allWeights = routerOutput.getAllWeights();
        int batchSize = allWeights.length;
        
        // 计算每个专家的重要性(importance)
        float[] importance = new float[numExperts];
        for (int b = 0; b < batchSize; b++) {
            for (int e = 0; e < numExperts; e++) {
                importance[e] += allWeights[b][e];
            }
        }
        
        // 归一化
        float importanceSum = 0.0f;
        for (float imp : importance) {
            importanceSum += imp;
        }
        if (importanceSum > 0) {
            for (int e = 0; e < numExperts; e++) {
                importance[e] /= importanceSum;
            }
        }
        
        // 计算每个专家的负载(load)
        int[][] topKIndices = routerOutput.getTopKIndices();
        float[] load = new float[numExperts];
        
        for (int b = 0; b < batchSize; b++) {
            for (int k = 0; k < topK; k++) {
                int expertIdx = topKIndices[b][k];
                load[expertIdx] += 1.0f;
            }
        }
        
        // 归一化
        float loadSum = 0.0f;
        for (float ld : load) {
            loadSum += ld;
        }
        if (loadSum > 0) {
            for (int e = 0; e < numExperts; e++) {
                load[e] /= loadSum;
            }
        }
        
        return new LoadBalanceStats(importance, load);
    }
    
    /**
     * 获取专家使用统计
     */
    public ExpertUsageStats getUsageStats() {
        float[] usageRate = new float[numExperts];
        
        if (totalCalls > 0) {
            for (int e = 0; e < numExperts; e++) {
                usageRate[e] = (float) expertUsageCount[e] / totalCalls;
            }
        }
        
        return new ExpertUsageStats(expertUsageCount.clone(), usageRate, totalCalls);
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        expertUsageCount = new long[numExperts];
        totalCalls = 0;
    }
    
    /**
     * 获取Router
     */
    public ExpertRouter getRouter() {
        return router;
    }
    
    /**
     * 获取专家列表
     */
    public List<ExpertNetwork> getExperts() {
        return new ArrayList<>(experts);
    }
    
    /**
     * 获取专家数量
     */
    public int getNumExperts() {
        return numExperts;
    }
    
    /**
     * 获取Top-K数量
     */
    public int getTopK() {
        return topK;
    }
    
    @Override
    public String toString() {
        return String.format("MoEBlock(in=%d, hidden=%d, out=%d, experts=%d, topK=%d)",
            inputDim, hiddenDim, outputDim, numExperts, topK);
    }
    
    /**
     * 负载均衡统计
     */
    public static class LoadBalanceStats {
        private final float[] importance;  // 专家重要性
        private final float[] load;        // 专家负载
        
        public LoadBalanceStats(float[] importance, float[] load) {
            this.importance = importance;
            this.load = load;
        }
        
        public float[] getImportance() {
            return importance;
        }
        
        public float[] getLoad() {
            return load;
        }
    }
    
    /**
     * 专家使用统计
     */
    public static class ExpertUsageStats {
        private final long[] usageCount;   // 使用次数
        private final float[] usageRate;   // 使用率
        private final long totalCalls;     // 总调用次数
        
        public ExpertUsageStats(long[] usageCount, float[] usageRate, long totalCalls) {
            this.usageCount = usageCount;
            this.usageRate = usageRate;
            this.totalCalls = totalCalls;
        }
        
        public long[] getUsageCount() {
            return usageCount;
        }
        
        public float[] getUsageRate() {
            return usageRate;
        }
        
        public long getTotalCalls() {
            return totalCalls;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("ExpertUsageStats{\n");
            for (int i = 0; i < usageCount.length; i++) {
                sb.append(String.format("  Expert%d: count=%d, rate=%.2f%%\n",
                    i, usageCount[i], usageRate[i] * 100));
            }
            sb.append(String.format("  Total calls: %d\n}", totalCalls));
            return sb.toString();
        }
    }
}