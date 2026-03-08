package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.TaskType;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3混合专家层(Mixture of Experts Layer)
 * 
 * 核心创新：通过门控网络动态选择Top-K个路由专家处理输入，实现参数高效和任务专门化。
 * 
 * 组件：
 * 1. 共享专家(Shared Experts)    - 每次必激活，提供通用知识（DeepSeekMoE创新）
 * 2. 门控网络(Gating Network)    - 计算路由专家的选择概率
 * 3. 路由专家(Routed Experts)    - Top-K选择的独立前馈网络
 * 4. 加权组合                       - 共享专家输出 + Top-K路由专家加权输出
 * 
 * 架构：
 * Input → Shared Experts → 输出直接累加
 *      → Gating Network → Top-K Selection → Routed Experts → 加权组合
 * 输出 = Shared输出 + Routed输出
 * 
 * ExpertNetwork FFN结构（SwiGLU）：
 * output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
 * 
 * @author leavesfly
 * @version 2.0
 */
public class DeepSeekV3MoELayer extends Module {

    /** 负载均衡损失为零的判断阈值 */
    private static final float ZERO_THRESHOLD = 1e-9f;

    /** 任务类型总数（用于任务感知路由偏置分配） */
    private static final int NUM_TASK_TYPES = 5;

    private final DeepSeekV3Config config;

    // 门控网络
    private Linear gatingNetwork;

    // 共享专家列表（每次必激活，DeepSeekMoE核心创新）
    private List<ExpertNetwork> sharedExperts;

    // 路由专家列表（Top-K选择）
    private List<ExpertNetwork> routedExperts;

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
        
        // 2. 初始化共享专家（每次必激活）
        sharedExperts = new ArrayList<>();
        for (int i = 0; i < config.getNumSharedExperts(); i++) {
            ExpertNetwork shared = new ExpertNetwork(
                name + "_shared_expert_" + i,
                config.getNEmbd(),
                config.getExpertHiddenDim()
            );
            sharedExperts.add(shared);
            registerModule("shared_expert_" + i, shared);
        }
        
        // 3. 初始化路由专家（Top-K选择）
        routedExperts = new ArrayList<>();
        for (int i = 0; i < config.getNumExperts(); i++) {
            ExpertNetwork expert = new ExpertNetwork(
                name + "_expert_" + i,
                config.getNEmbd(),
                config.getExpertHiddenDim()
            );
            routedExperts.add(expert);
            registerModule("expert_" + i, expert);
        }
        
        // 4. 初始化Dropout层
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
        // 执行MoE计算，不携带任务类型（标准路由模式）
        MoEOutput moeOutput = computeMoE(inputs[0], null);
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
        
        // 3. 计算门控概率（softmax归一化）
        Variable gatingProbs = gatingLogits.softMax();
        
        // 4. Top-K选择
        TopKResult topKResult = selectTopK(gatingProbs, config.getTopK());
        
        // 5. 共享专家计算（每次必激活）
        Variable sharedOutput = computeSharedExpertsOutput(input);
        
        // 6. 路由专家加权组合
        Variable routedOutput = computeExpertOutputs(input, topKResult);
        
        // 7. 共享专家输出 + 路由专家输出
        Variable expertOutputs = sharedOutput.add(routedOutput);
        
        // 8. 计算负载均衡损失
        double loadBalanceLoss = computeLoadBalanceLoss(gatingProbs);
        
        return new MoEOutput(expertOutputs, gatingProbs, topKResult, loadBalanceLoss);
    }
    
    /**
     * 应用任务感知偏置：将任务特定偏置广播加到门控logits上
     */
    private Variable applyTaskAwareBias(Variable gatingLogits, TaskType taskType) {
        float[] taskBias = getTaskBias(taskType);
        // 扩展为 [1, 1, numExperts] 以支持广播加法
        Variable bias3D = new Variable(NdArray.of(taskBias))
            .reshape(Shape.of(1, 1, config.getNumExperts()));
        return gatingLogits.add(bias3D);
    }
    
    /**
     * 根据任务类型生成专家偏置向量
     * 使用取模运算确保索引不越界，每种任务类型均匀分配偏向的专家范围
     */
    private float[] getTaskBias(TaskType taskType) {
        int numExperts = config.getNumExperts();
        float[] bias = new float[numExperts];
        int taskId = taskType.getId();
        int expertsPerTask = Math.max(1, numExperts / NUM_TASK_TYPES);
        int startIdx = (taskId * expertsPerTask) % numExperts;
        for (int i = 0; i < expertsPerTask; i++) {
            bias[(startIdx + i) % numExperts] = 1.0f;
        }
        return bias;
    }
    
    /**
     * 计算共享专家输出（每次必激活，DeepSeekMoE核心创新）
     * 
     * @param input 输入张量 [batch_size, seq_len, nEmbd]
     * @return 共享专家输出的堆叠和
     */
    private Variable computeSharedExpertsOutput(Variable input) {
        if (sharedExperts.isEmpty()) {
            // 无共享专家时返回零张量
            NdArray zeros = NdArray.zeros(input.getValue().getShape());
            return new Variable(zeros);
        }
        // 所有共享专家输出直接相加（不需要编号，易于理解）
        Variable sharedOut = sharedExperts.get(0).forward(input);
        for (int i = 1; i < sharedExperts.size(); i++) {
            sharedOut = sharedOut.add(sharedExperts.get(i).forward(input));
        }
        return sharedOut;
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
     * 计算所有路由专家的输出并按 TopK 权重加权组合
     * 策略：让所有专家批量处理整个 batch，再根据 TopK 权重做稀疏累加，保证梯度回传
     */
    private Variable computeExpertOutputs(Variable input, TopKResult topKResult) {
        Shape inputShape = input.getValue().getShape();
        int batchSize = inputShape.getDimension(0);
        int seqLen    = inputShape.getDimension(1);
        int nEmbd     = inputShape.getDimension(2);

        // 所有路由专家批量计算各自输出
        List<Variable> allExpertOutputs = new ArrayList<>();
        for (ExpertNetwork expert : routedExperts) {
            allExpertOutputs.add(expert.forward(input));
        }

        return createWeightedExpertCombination(allExpertOutputs, topKResult, batchSize, seqLen, nEmbd);
    }
    
    /**
     * 根据 TopK 结果对各专家输出进行稀疏加权累加
     * 权重矩阵形状为 [batch, seq, 1]，通过广播与 [batch, seq, nEmbd] 相乘
     * 未被任何位置选中的专家直接跳过，减少无效计算
     */
    private Variable createWeightedExpertCombination(
            List<Variable> expertOutputs,
            TopKResult topKResult,
            int batchSize,
            int seqLen,
            int nEmbd) {

        Variable output = new Variable(NdArray.zeros(Shape.of(batchSize, seqLen, nEmbd)));

        for (int expertIdx = 0; expertIdx < expertOutputs.size(); expertIdx++) {
            Variable weightMask = createExpertWeightMask(expertIdx, topKResult, batchSize, seqLen);
            if (isZeroMask(weightMask)) {
                continue;
            }
            // 广播乘法: [batch, seq, 1] × [batch, seq, nEmbd]
            output = output.add(expertOutputs.get(expertIdx).mul(weightMask));
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
     * 检查权重 mask 是否全为零（未被任何 token 选中）
     */
    private boolean isZeroMask(Variable mask) {
        float sum = mask.getValue().sum().getNumber().floatValue();
        return Math.abs(sum) < ZERO_THRESHOLD;
    }
    
    /**
     * 计算负载均衡损失：以各专家平均激活频率与理想均匀分布的方差衡量不均衡程度
     * 方差越小说明专家利用越均匀，理想值为 0（所有专家频率均为 1/numExperts）
     */
    private double computeLoadBalanceLoss(Variable gatingProbs) {
        NdArray probsArray = gatingProbs.getValue();
        int batchSize  = probsArray.getShape().getDimension(0);
        int seqLen     = probsArray.getShape().getDimension(1);
        int numExperts = probsArray.getShape().getDimension(2);
        int totalTokens = batchSize * seqLen;

        // 统计每个专家跨所有 token 的平均激活频率
        float[] expertFreq = new float[numExperts];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int e = 0; e < numExperts; e++) {
                    expertFreq[e] += probsArray.get(b, t, e);
                }
            }
        }

        // 计算与均匀分布（1/numExperts）之间的方差
        float idealFreq = 1.0f / numExperts;
        float variance  = 0.0f;
        for (int e = 0; e < numExperts; e++) {
            float diff = (expertFreq[e] / totalTokens) - idealFreq;
            variance += diff * diff;
        }

        return variance * config.getLoadBalanceLossWeight();
    }
    
    /**
     * 专家网络内部类（SwiGLU FFN）
     * 
     * SwiGLU结构（对标DeepSeek-V3论文）：
     * output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
     * 
     * 相比GELU-FFN的改进：
     * - 门控机制过滤信息，训练更稳定
     * - SiLU激活函数比GELU计算简单且效果相当
     */
    private static class ExpertNetwork extends Module {
        private final Linear gateProj;   // 门控投影: inputDim -> hiddenDim
        private final SiLU   silu;       // SiLU激活函数
        private final Linear upProj;     // 上投影: inputDim -> hiddenDim
        private final Linear downProj;   // 下投影: hiddenDim -> inputDim
        
        public ExpertNetwork(String name, int inputDim, int hiddenDim) {
            super(name);
            
            // 门控分支: inputDim -> hiddenDim
            gateProj = new Linear(name + "_gate", inputDim, hiddenDim, true);
            registerModule("gate", gateProj);
            
            // SiLU激活
            silu = new SiLU(name + "_silu");
            registerModule("silu", silu);
            
            // 上分支: inputDim -> hiddenDim
            upProj = new Linear(name + "_up", inputDim, hiddenDim, true);
            registerModule("up", upProj);
            
            // 下投影: hiddenDim -> inputDim
            downProj = new Linear(name + "_down", hiddenDim, inputDim, true);
            registerModule("down", downProj);
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable x = inputs[0];
            // SwiGLU: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
            Variable gate = silu.forward(gateProj.forward(x));
            Variable up = upProj.forward(x);
            Variable hidden = gate.mul(up);
            return downProj.forward(hidden);
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
