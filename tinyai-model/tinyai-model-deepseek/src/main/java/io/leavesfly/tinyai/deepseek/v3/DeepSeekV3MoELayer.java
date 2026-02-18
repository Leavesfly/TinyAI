package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3混合专家层(Mixture of Experts Layer)
 * 
 * 核心创新：通过门控网络动态选择Top-K个专家处理输入，实现参数高效和任务专门化。
 * 
 * 组件：
 * 1. 门控网络(Gating Network) - 计算每个专家的选择概率
 * 2. 专家网络(Expert Networks) - 8个独立的前馈网络
 * 3. Top-K选择 - 选择概率最高的K个专家
 * 4. 加权组合 - 根据门控权重组合专家输出
 * 
 * 架构：
 * Input → Gating Network → Top-K Selection → Expert Processing → Weighted Combination → Output
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3MoELayer extends Module {
    
    private final DeepSeekV3Config config;
    
    // 门控网络
    private Linear gatingNetwork;
    
    // 专家网络列表
    private List<ExpertNetwork> experts;
    
    // Dropout层
    private Dropout expertDropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3MoELayer(String name, DeepSeekV3Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化组件
     */
    private void initializeComponents() {
        // 1. 初始化门控网络: nEmbd -> numExperts
        gatingNetwork = new Linear(
            name + "_gating",
            config.getNEmbd(),
            config.getNumExperts(),
            true  // 使用偏置
        );
        registerModule("gating", gatingNetwork);
        
        // 2. 初始化专家网络
        experts = new ArrayList<>();
        for (int i = 0; i < config.getNumExperts(); i++) {
            ExpertNetwork expert = new ExpertNetwork(
                name + "_expert_" + i,
                config.getNEmbd(),
                config.getExpertHiddenDim()
            );
            experts.add(expert);
            registerModule("expert_" + i, expert);
        }
        
        // 3. 初始化Dropout层
        expertDropout = new Dropout(
            name + "_expert_dropout",
            (float) config.getExpertDropout()
        );
        registerModule("expert_dropout", expertDropout);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入张量 [batch_size, seq_len, nEmbd]
     *               inputs[1](可选)为任务类型 TaskType
     * @return MoE输出结果
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        TaskType taskType = null;
        if (inputs.length > 1 && inputs[1] != null) {
            // 这里假设inputs[1]包含任务类型信息
            // 实际使用中需要从Variable中提取TaskType
        }
        
        // 执行MoE计算
        MoEOutput moeOutput = computeMoE(input, taskType);
        
        // 应用dropout
        return expertDropout.forward(moeOutput.output);
    }
    
    /**
     * 执行MoE计算（核心方法）
     * 
     * @param input 输入张量 [batch_size, seq_len, nEmbd]
     * @param taskType 任务类型（可选，用于任务感知路由）
     * @return MoE输出结果
     */
    public MoEOutput computeMoE(Variable input, TaskType taskType) {

        // 1. 计算门控logits: [batch_size, seq_len, numExperts]
        Variable gatingLogits = gatingNetwork.forward(input);
        
        // 2. 应用任务感知偏置（如果提供了任务类型）
        if (taskType != null && config.isEnableTaskAwareRouting()) {
            gatingLogits = applyTaskAwareBias(gatingLogits, taskType);
        }
        
        // 3. 计算门控概率（softmax）
        Variable gatingProbs = softmax(gatingLogits, -1);
        
        // 4. Top-K选择
        TopKResult topKResult = selectTopK(gatingProbs, config.getTopK());
        
        // 5. 专家计算
        Variable expertOutputs = computeExpertOutputs(input, topKResult);
        
        // 6. 计算负载均衡损失
        double loadBalanceLoss = computeLoadBalanceLoss(gatingProbs);
        
        return new MoEOutput(expertOutputs, gatingProbs, topKResult, loadBalanceLoss);
    }
    
    /**
     * 应用任务感知偏置 (使用Variable算子)
     */
    private Variable applyTaskAwareBias(Variable gatingLogits, TaskType taskType) {
        // ✅ 使用Variable算子添加偏置
        // 获取任务偏置并转换为Variable
        float[] taskBias = getTaskBias(taskType);
        
        // 将偏置扩展为 [1, 1, numExperts] 形状
        NdArray biasArray = NdArray.of(taskBias);
        Variable biasVar = new Variable(biasArray);
        Variable bias3D = biasVar.reshape(Shape.of(1, 1, config.getNumExperts()));
        
        // 使用Variable的add算子（会自动广播）
        return gatingLogits.add(bias3D);
    }
    
    /**
     * 获取任务类型的专家偏置
     * 
     * 根据实际专家数量动态分配偏置，避免越界
     */
    private float[] getTaskBias(TaskType taskType) {
        int numExperts = config.getNumExperts();
        float[] bias = new float[numExperts];
        
        // 根据任务类型和实际专家数量动态分配偏置
        // 使用取模运算确保索引不越界
        int taskId = taskType.getId();
        
        // 每个任务类型倾向于2个专家（如果专家数足够）
        int expertsPerTask = Math.max(1, numExperts / 5);  // 5种任务类型
        int startIdx = (taskId * expertsPerTask) % numExperts;
        
        for (int i = 0; i < expertsPerTask && i < numExperts; i++) {
            int expertIdx = (startIdx + i) % numExperts;
            bias[expertIdx] = 1.0f;
        }
        
        return bias;
    }
    
    /**
     * Softmax激活函数 (使用Variable算子)
     */
    private Variable softmax(Variable logits, int dim) {
        // ✅ 直接使用Variable的softMax算子
        return logits.softMax();
    }
    
    /**
     * 选择Top-K专家
     */
    private TopKResult selectTopK(Variable probs, int k) {
        NdArray probsArray = probs.getValue();
        int batchSize = probsArray.getShape().getDimension(0);
        int seqLen = probsArray.getShape().getDimension(1);
        int numExperts = probsArray.getShape().getDimension(2);
        
        int[][][] topKIndices = new int[batchSize][seqLen][k];
        float[][][] topKWeights = new float[batchSize][seqLen][k];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 获取当前位置的所有专家概率
                float[] expertProbs = new float[numExperts];
                for (int e = 0; e < numExperts; e++) {
                    expertProbs[e] = probsArray.get(b, t, e);
                }
                
                // 选择Top-K
                int[] topK = getTopKIndices(expertProbs, k);
                float[] topKProbs = new float[k];
                float sumTopKProbs = 0.0f;
                
                for (int i = 0; i < k; i++) {
                    topKProbs[i] = expertProbs[topK[i]];
                    sumTopKProbs += topKProbs[i];
                }
                
                // 归一化权重（使Top-K概率和为1）
                for (int i = 0; i < k; i++) {
                    topKIndices[b][t][i] = topK[i];
                    topKWeights[b][t][i] = topKProbs[i] / sumTopKProbs;
                }
            }
        }
        
        return new TopKResult(topKIndices, topKWeights);
    }
    
    /**
     * 获取Top-K索引
     */
    private int[] getTopKIndices(float[] values, int k) {
        int[] indices = new int[k];
        boolean[] used = new boolean[values.length];
        
        for (int i = 0; i < k; i++) {
            int maxIdx = -1;
            float maxVal = Float.NEGATIVE_INFINITY;
            
            for (int j = 0; j < values.length; j++) {
                if (!used[j] && values[j] > maxVal) {
                    maxVal = values[j];
                    maxIdx = j;
                }
            }
            
            indices[i] = maxIdx;
            used[maxIdx] = true;
        }
        
        return indices;
    }
    
    /**
     * 计算所有专家的输出并加权组合
     * 
     * ✅ 优化方案：批量计算，完全在Variable层面
     * 策略：
     * 1. 让所有专家并行处理整个batch的输入
     * 2. 根据TopK结果构建权重矩阵
     * 3. 使用Variable算子进行加权组合
     * 
     * 这样既保证了梯度回传，又避免了逐位置的循环
     */
    private Variable computeExpertOutputs(Variable input, TopKResult topKResult) {
        int batchSize = input.getValue().getShape().getDimension(0);
        int seqLen = input.getValue().getShape().getDimension(1);
        int nEmbd = input.getValue().getShape().getDimension(2);
        int numExperts = config.getNumExperts();
        
        // ✅ 方案1：所有专家并行计算（推荐用于推理）
        // 存储所有专家的输出
        List<Variable> expertOutputs = new ArrayList<>();
        for (int i = 0; i < numExperts; i++) {
            // 每个专家处理整个batch
            Variable expertOut = experts.get(i).forward(input);
            expertOutputs.add(expertOut);
        }
        
        // ✅ 构建专家选择和权重矩阵
        // 对于每个位置(b,t)，根据TopK结果加权组合对应的专家输出
        Variable result = createWeightedExpertCombination(
            expertOutputs, topKResult, batchSize, seqLen, nEmbd
        );
        
        return result;
    }
    
    /**
     * ✅ 根据TopK结果加权组合专家输出（在Variable层面）
     */
    private Variable createWeightedExpertCombination(
            List<Variable> expertOutputs,
            TopKResult topKResult,
            int batchSize,
            int seqLen,
            int nEmbd) {
        
        // 初始化输出为零
        NdArray outputArray = NdArray.zeros(Shape.of(batchSize, seqLen, nEmbd));
        Variable output = new Variable(outputArray);
        
        // 对每个专家，构建其权重mask并累加
        for (int expertIdx = 0; expertIdx < expertOutputs.size(); expertIdx++) {
            // 构建该专家的权重矩阵 [batch_size, seq_len, 1]
            Variable weightMask = createExpertWeightMask(
                expertIdx, topKResult, batchSize, seqLen
            );
            
            // 如果该专家没有被任何位置选中，跳过
            if (isZeroMask(weightMask)) {
                continue;
            }
            
            // 获取该专家的输出并加权
            Variable expertOut = expertOutputs.get(expertIdx);
            
            // 使用广播乘法: [batch, seq, 1] × [batch, seq, nEmbd]
            // 避免使用repeat创建大量中间变量
            Variable weightedOut = expertOut.mul(weightMask);
            
            // ✅ 累加到输出（在Variable层面）
            output = output.add(weightedOut);
        }
        
        return output;
    }
    
    /**
     * 为指定专家创建权重mask
     * 返回 [batch_size, seq_len, 1] 的权重矩阵
     */
    private Variable createExpertWeightMask(
            int expertIdx,
            TopKResult topKResult,
            int batchSize,
            int seqLen) {
        
        float[][][] weights = new float[batchSize][seqLen][1];
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 检查该位置的TopK中是否包含当前专家
                for (int k = 0; k < config.getTopK(); k++) {
                    if (topKResult.indices[b][t][k] == expertIdx) {
                        weights[b][t][0] = topKResult.weights[b][t][k];
                        break;
                    }
                }
            }
        }
        
        return new Variable(NdArray.of(weights));
    }
    
    /**
     * 检查权重mask是否全为0
     */
    private boolean isZeroMask(Variable mask) {
        NdArray arr = mask.getValue();
        float sum = arr.sum().getNumber().floatValue();
        return Math.abs(sum) < 1e-9f;
    }
    
    /**
     * 计算负载均衡损失
     * 目标：确保所有专家被均匀使用
     */
    private double computeLoadBalanceLoss(Variable gatingProbs) {
        NdArray probsArray = gatingProbs.getValue();
        int batchSize = probsArray.getShape().getDimension(0);
        int seqLen = probsArray.getShape().getDimension(1);
        int numExperts = probsArray.getShape().getDimension(2);
        
        // 计算每个专家的平均使用频率
        float[] expertFreq = new float[numExperts];
        int totalTokens = batchSize * seqLen;
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int e = 0; e < numExperts; e++) {
                    expertFreq[e] += probsArray.get(b, t, e);
                }
            }
        }
        
        for (int e = 0; e < numExperts; e++) {
            expertFreq[e] /= totalTokens;
        }
        
        // 计算方差（理想情况下所有专家频率都接近1/numExperts）
        float idealFreq = 1.0f / numExperts;
        float variance = 0.0f;
        
        for (int e = 0; e < numExperts; e++) {
            float diff = expertFreq[e] - idealFreq;
            variance += diff * diff;
        }
        
        return variance * config.getLoadBalanceLossWeight();
    }
    
    /**
     * 专家网络内部类
     * 每个专家是一个独立的两层前馈网络
     */
    private static class ExpertNetwork extends Module {
        private Linear fc1;
        private GELU activation;
        private Linear fc2;
        
        public ExpertNetwork(String name, int inputDim, int hiddenDim) {
            super(name);
            
            // 第一层：inputDim -> hiddenDim
            fc1 = new Linear(name + "_fc1", inputDim, hiddenDim, true);
            registerModule("fc1", fc1);
            
            // 激活函数
            activation = new GELU(name + "_gelu");
            registerModule("gelu", activation);
            
            // 第二层：hiddenDim -> inputDim
            fc2 = new Linear(name + "_fc2", hiddenDim, inputDim, true);
            registerModule("fc2", fc2);
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            x = fc1.forward(x);
            x = activation.forward(x);
            x = fc2.forward(x);
            return x;
        }
    }
    
    /**
     * Top-K选择结果类
     */
    public static class TopKResult {
        public final int[][][] indices;   // [batch_size, seq_len, k]
        public final float[][][] weights; // [batch_size, seq_len, k]
        
        public TopKResult(int[][][] indices, float[][][] weights) {
            this.indices = indices;
            this.weights = weights;
        }
    }
    
    /**
     * MoE输出结果类
     */
    public static class MoEOutput {
        /** MoE层的输出 */
        public final Variable output;
        /** 所有专家的门控概率 */
        public final Variable gatingProbs;
        /** Top-K选择结果 */
        public final TopKResult topKResult;
        /** 负载均衡损失 */
        public final double loadBalanceLoss;
        
        public MoEOutput(Variable output, Variable gatingProbs, 
                        TopKResult topKResult, double loadBalanceLoss) {
            this.output = output;
            this.gatingProbs = gatingProbs;
            this.topKResult = topKResult;
            this.loadBalanceLoss = loadBalanceLoss;
        }
        
        @Override
        public String toString() {
            return String.format(
                "MoEOutput{loadBalanceLoss=%.6f, outputShape=%s}",
                loadBalanceLoss,
                output.getValue().getShape()
            );
        }
    }
}
