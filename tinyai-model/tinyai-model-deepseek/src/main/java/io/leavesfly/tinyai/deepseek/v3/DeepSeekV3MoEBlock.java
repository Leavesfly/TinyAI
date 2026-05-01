package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.deepseek.base.DeepSeekBaseConfig;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.core.Parameter;
import io.leavesfly.tinyai.nnet.layer.activation.SiLU;
import io.leavesfly.tinyai.nnet.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V3混合专家层(Mixture of Experts Layer)
 * <p>
 * 对标 DeepSeek-V3 论文的核心创新：
 * 1. Sigmoid 路由：用 Sigmoid 替代 Softmax，每个专家独立计算激活概率
 * 2. 无辅助损失负载均衡：通过可学习的 bias 向量动态调节专家选择
 * - 路由分数 = sigmoid(gate(x)) + bias（bias 仅影响 Top-K 选择，不影响最终权重）
 * - 训练时根据专家负载动态更新 bias（过载专家降低 bias，欠载专家提高 bias）
 * 3. Top-K 选择后，权重用 sigmoid 值（不加 bias）进行归一化
 * <p>
 * 组件：
 * 1. 共享专家(Shared Experts)    - 每次必激活，提供通用知识（DeepSeekMoE创新）
 * 2. 门控网络(Gating Network)    - 计算路由 logits
 * 3. 路由专家(Routed Experts)    - Top-K选择的独立前馈网络（SwiGLU FFN）
 * 4. 专家偏置(Expert Bias)       - 无辅助损失负载均衡的可学习参数
 * <p>
 * 架构：
 * Input → Shared Experts → 输出直接累加
 * → Gating Network → Sigmoid + Bias → Top-K Selection → Routed Experts → 加权组合
 * 输出 = Shared输出 + Routed输出
 * <p>
 * ExpertNetwork FFN结构（SwiGLU）：
 * output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
 *
 * @author leavesfly
 * @version 3.0
 */
public class DeepSeekV3MoEBlock extends Module {

    /**
     * 权重 mask 为零的判断阈值
     */
    private static final float ZERO_THRESHOLD = 1e-9f;

    /**
     * bias 动态更新步长（无辅助损失负载均衡）
     */
    private static final float BIAS_UPDATE_SPEED = 0.001f;

    private final DeepSeekBaseConfig config;

    // 门控网络: nEmbd -> numExperts
    private Linear gatingNetwork;

    // 专家偏置向量（无辅助损失负载均衡的核心参数）
    // bias 仅影响 Top-K 选择，不参与最终权重计算
    private Parameter expertBias;

    // 共享专家列表（每次必激活，DeepSeekMoE核心创新）
    private List<ExpertNetwork> sharedExperts;

    // 路由专家列表（Top-K选择）
    private List<ExpertNetwork> routedExperts;

    // Dropout层
    private Dropout expertDropout;

    /**
     * 最近一次 forward 的专家选择统计（用于 optimizer.update 后延迟更新 expertBias）
     * 这样能保证 forward 过程中 bias 值保持稳定，避免同一次前向/反向传播中 bias 被"读完又写"。
     */
    private int[] lastSelectionCount;
    private int lastTotalTokens;

    /**
     * 构造函数
     *
     * @param name   模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3MoEBlock(String name, DeepSeekBaseConfig config) {
        super(name);
        this.config = config;
        initializeComponents();
    }

    /**
     * 初始化组件
     */
    private void initializeComponents() {
        // 1. 初始化门控网络: nEmbd -> numExperts（不使用偏置，bias 由 expertBias 单独管理）
        gatingNetwork = new Linear(
                name + "_gating",
                config.getNEmbd(),
                config.getNumExperts(),
                false  // 不使用偏置，路由 bias 由 expertBias 参数独立管理
        );
        registerModule("gating", gatingNetwork);

        // 2. 初始化专家偏置向量（初始化为 0，训练时动态更新以实现负载均衡）
        expertBias = new Parameter(NdArray.zeros(Shape.of(config.getNumExperts())), false);
        registerParameter("expert_bias", expertBias);

        // 3. 初始化共享专家（每次必激活）
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

        // 4. 初始化路由专家（Top-K选择）
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

        // 5. 初始化Dropout层
        expertDropout = new Dropout(
                name + "_expert_dropout",
                (float) config.getExpertDropout()
        );
        registerModule("expert_dropout", expertDropout);
    }

    /**
     * 前向传播（标准 Sigmoid 路由，无任务感知）
     *
     * <p><b>注意：</b>当前实现 <b>不支持</b> 任务感知路由。配置项 {@code config.enableTaskAwareRouting} 与
     * {@code taskEmbedDim}/{@code taskClassifierHiddenDim}/{@code numTaskTypes} 仅保留为元数据占位，
     * 不会真正作用于 gating 计算。如需任务感知，请参考论文 DeepSeekMoE 自行扩展 {@link #computeMoE(Variable)}：
     * 将任务 embedding 与 gating logits 融合即可，但会破坏与预训练权重的兼容性。
     *
     * <p>历史 javadoc 曾声称 inputs[1] 可传 TaskType，属于<b>文档与实现不一致</b>的过期描述，已删除。
     *
     * @param inputs inputs[0] 为输入张量 [batch_size, seq_len, nEmbd]
     * @return MoE 输出（经 dropout 后的 Variable）
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        // 执行 MoE 计算（当前仅支持标准路由模式，inputs[1] 之后的参数会被忽略）
        MoEOutput moeOutput = computeMoE(inputs[0]);
        // 应用 dropout
        return expertDropout.forward(moeOutput.output);
    }

    /**
     * 执行MoE计算（核心方法）
     * <p>
     * 对标 DeepSeek-V3 论文的 Sigmoid 路由 + 无辅助损失负载均衡：
     * 1. 计算门控 logits → sigmoid 得到每个专家的独立激活概率
     * 2. 路由分数 = sigmoid(logits) + bias（bias 仅影响 Top-K 选择）
     * 3. Top-K 选择后，权重用 sigmoid 值（不加 bias）进行归一化
     * 4. 根据专家负载动态更新 bias（无辅助损失负载均衡）
     *
     * @param input 输入张量 [batch_size, seq_len, nEmbd]
     * @return MoE输出结果
     */
    public MoEOutput computeMoE(Variable input) {

        // 1. 计算门控 logits: [batch_size, seq_len, numExperts]
        Variable gatingLogits = gatingNetwork.forward(input);

        // 2. Sigmoid 激活（替代 Softmax，每个专家独立计算概率）
        Variable sigmoidProbs = gatingLogits.sigmoid();

        // 3. 路由分数 = sigmoid(logits) + bias（bias 仅影响 Top-K 选择）
        //    bias 广播为 [1, 1, numExperts] 以匹配 [batch, seq, numExperts]
        Variable biasExpanded = expertBias.reshape(Shape.of(1, 1, config.getNumExperts()));
        Variable routingScores = sigmoidProbs.add(biasExpanded);

        // 4. Top-K 选择（基于 routingScores），但权重用 sigmoidProbs（不含 bias）
        TopKResult topKResult = selectTopKWithSeparateWeights(routingScores, sigmoidProbs, config.getTopK());

        // 5. 共享专家计算（每次必激活）
        Variable sharedOutput = computeSharedExpertsOutput(input);

        // 6. 路由专家加权组合（传入 sigmoidProbs 保持计算图连通）
        Variable routedOutput = computeExpertOutputs(input, topKResult, sigmoidProbs);

        // 7. 共享专家输出 + 路由专家输出
        Variable expertOutputs = sharedOutput.add(routedOutput);

        // 8. 记录专家选择统计（真正的 bias 更新延迟到 optimizer.update() 之后执行，
        //    由训练循环或模型主动调用 updateExpertBiasAfterStep()。
        //    这样可以保证同一次 forward+backward 期间 bias 值保持不变，不污染自动微分。）
        recordSelectionCount(topKResult);

        return new MoEOutput(expertOutputs, sigmoidProbs, topKResult, 0.0);
    }

    /**
     * 记录最近一次 forward 的专家选择统计（供 updateExpertBiasAfterStep 使用）
     */
    private void recordSelectionCount(TopKResult topKResult) {
        int numExperts = config.getNumExperts();
        int batchSize = topKResult.indices.length;
        int seqLen = topKResult.indices[0].length;
        int topK = config.getTopK();

        int[] count = new int[numExperts];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int ki = 0; ki < topK; ki++) {
                    count[topKResult.indices[b][t][ki]]++;
                }
            }
        }
        this.lastSelectionCount = count;
        this.lastTotalTokens = batchSize * seqLen;
    }

    /**
     * 在优化器 step 之后更新专家 bias（无辅助损失负载均衡）
     * <p>
     * 对标 DeepSeek-V3 论文的核心创新：
     * - 过载专家（频率高于均匀分布）：降低 bias，使其更难被选中
     * - 欠载专家（频率低于均匀分布）：提高 bias，使其更容易被选中
     * <p>
     * 训练循环应在 optimizer.update() 之后调用本方法。这样做的好处：
     * 1. 不会在 forward 内部修改 bias，避免和自动微分产生交互
     * 2. bias 更新与主优化器 step 同频率，语义清晰
     * 3. 推理时（model.eval()）默认跳过，不会污染推理结果
     *
     * 注：如未调用过 forward，本方法不执行任何操作。
     */
    public void updateExpertBiasAfterStep() {
        if (lastSelectionCount == null || lastTotalTokens <= 0) {
            return;
        }
        int numExperts = config.getNumExperts();
        int topK = config.getTopK();
        float idealCount = (float) lastTotalTokens * topK / numExperts;

        // NdArrayCpu.getArray() 返回底层 float[] 的直接引用（见 NdArrayCpu#getArray），
        // 因此原地修改 biasValues 即可直接更新 expertBias 的内容。
        // 因 expertBias.requireGrad=false，此写入不会产生梯度副作用。
        NdArray biasData = expertBias.data();
        float[] biasValues = biasData.getArray();
        for (int e = 0; e < numExperts; e++) {
            float deviation = lastSelectionCount[e] - idealCount;
            biasValues[e] -= BIAS_UPDATE_SPEED * deviation;
        }

        // 用完即清，避免同一次统计被错误地多次应用（例如多次 optimizer.update 只 forward 一次的异常情况）
        lastSelectionCount = null;
        lastTotalTokens = 0;
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
     * 选择 Top-K 专家（Sigmoid 路由版本）
     * <p>
     * 对标 DeepSeek-V3 论文：
     * - 基于 routingScores（sigmoid + bias）通过 topk 算子选择 Top-K 专家索引
     * - 但最终权重使用 sigmoidProbs（不含 bias），确保 bias 不影响梯度
     * - Top-K 权重归一化使概率和为 1
     *
     * @param routingScores sigmoid(logits) + bias，用于 Top-K 选择 [batch, seq, numExperts]
     * @param sigmoidProbs  sigmoid(logits)，用于计算最终权重 [batch, seq, numExperts]
     * @param k             选择的专家数量
     * @return Top-K 选择结果（索引和归一化权重）
     */
    private TopKResult selectTopKWithSeparateWeights(Variable routingScores, Variable sigmoidProbs, int k) {
        // 1. 使用 topk 算子基于 routingScores（含 bias）选择 Top-K 专家索引
        NdArray[] topkResult = routingScores.getValue().topk(k);
        NdArray topkIndicesArray = topkResult[0];  // [..., K]，索引以 float 存储

        // 2. 用 topk 索引从 sigmoidProbs（不含 bias）中取值并归一化
        NdArray probsArray = sigmoidProbs.getValue();
        int batchSize = probsArray.getShape().getDimension(0);
        int seqLen = probsArray.getShape().getDimension(1);

        int[][][] topKIndices = new int[batchSize][seqLen][k];
        float[][][] topKWeights = new float[batchSize][seqLen][k];

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                // 从 topk 结果中取出索引，并用 sigmoidProbs 计算归一化权重
                float sumProbs = 0.0f;
                for (int i = 0; i < k; i++) {
                    int expertIdx = (int) topkIndicesArray.get(b, t, i);
                    topKIndices[b][t][i] = expertIdx;
                    sumProbs += probsArray.get(b, t, expertIdx);
                }

                for (int i = 0; i < k; i++) {
                    float prob = probsArray.get(b, t, topKIndices[b][t][i]);
                    topKWeights[b][t][i] = sumProbs > ZERO_THRESHOLD ? prob / sumProbs : 1.0f / k;
                }
            }
        }

        return new TopKResult(topKIndices, topKWeights);
    }

    /**
     * 仅计算被 Top-K 选中的路由专家输出并加权组合（延迟计算优化）
     * <p>
     * 优化：先收集哪些专家被选中，仅对被选中的专家执行 forward，
     * 避免未被选中的专家做无效计算，减少约 (1 - topK/numExperts) 的计算量。
     * <p>
     * 计算图保持：权重掩码通过 sigmoidProbs 的 Variable 操作构建，
     * 确保门控网络的梯度可以通过加权组合回传。
     *
     * @param input        输入张量 [batch_size, seq_len, nEmbd]
     * @param topKResult   Top-K 选择结果（索引和归一化权重，用于确定选中哪些专家）
     * @param sigmoidProbs 门控网络的 sigmoid 概率 [batch_size, seq_len, numExperts]（保持计算图）
     * @return 路由专家的加权组合输出
     */
    private Variable computeExpertOutputs(Variable input, TopKResult topKResult, Variable sigmoidProbs) {
        Shape inputShape = input.getValue().getShape();
        int batchSize = inputShape.getDimension(0);
        int seqLen = inputShape.getDimension(1);
        int nEmbd = inputShape.getDimension(2);

        // 收集被选中的专家集合
        boolean[] selectedExperts = new boolean[routedExperts.size()];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int ki = 0; ki < config.getTopK(); ki++) {
                    selectedExperts[topKResult.indices[b][t][ki]] = true;
                }
            }
        }

        // 仅对被选中的专家执行 forward（延迟计算）
        Variable[] expertOutputCache = new Variable[routedExperts.size()];
        for (int e = 0; e < routedExperts.size(); e++) {
            if (selectedExperts[e]) {
                expertOutputCache[e] = routedExperts.get(e).forward(input);
            }
        }

        // 加权组合（通过 sigmoidProbs 的 Variable 操作保持计算图连通）
        Variable output = new Variable(NdArray.zeros(Shape.of(batchSize, seqLen, nEmbd)));
        for (int expertIdx = 0; expertIdx < routedExperts.size(); expertIdx++) {
            if (!selectedExperts[expertIdx]) {
                continue;
            }
            // 从 sigmoidProbs 中通过 Variable 操作提取该专家的权重（保持计算图）
            Variable weightMask = createExpertWeightMaskFromGraph(expertIdx, topKResult, sigmoidProbs, batchSize, seqLen);
            if (weightMask == null) {
                continue;
            }
            output = output.add(expertOutputCache[expertIdx].mul(weightMask));
        }

        return output;
    }

    /**
     * 从 sigmoidProbs 的计算图中构建专家权重掩码（保持梯度流）
     * <p>
     * 通过 Variable.select 从 sigmoidProbs 中提取指定专家的概率，
     * 再根据 Top-K 选择结果构建归一化的权重掩码。
     * 整个过程基于 Variable 操作，梯度可以从权重回传到门控网络。
     *
     * @param expertIdx    专家索引
     * @param topKResult   Top-K 选择结果
     * @param sigmoidProbs 门控网络的 sigmoid 概率 [batch_size, seq_len, numExperts]
     * @param batchSize    批次大小
     * @param seqLen       序列长度
     * @return 权重掩码 [batch_size, seq_len, 1]，如果该专家未被任何 token 选中则返回 null
     */
    private Variable createExpertWeightMaskFromGraph(
            int expertIdx,
            TopKResult topKResult,
            Variable sigmoidProbs,
            int batchSize,
            int seqLen) {

        // 构建选择掩码：标记哪些位置选中了该专家
        float[][][] selectionMask = new float[batchSize][seqLen][1];
        boolean hasSelection = false;
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int k = 0; k < config.getTopK(); k++) {
                    if (topKResult.indices[b][t][k] == expertIdx) {
                        selectionMask[b][t][0] = 1.0f;
                        hasSelection = true;
                        break;
                    }
                }
            }
        }
        if (!hasSelection) {
            return null;
        }

        // 从 sigmoidProbs 中提取该专家的概率（保持计算图）
        // sigmoidProbs: [batch, seq, numExperts] → select dim=2, index=expertIdx → [batch, seq]
        Variable expertProb = sigmoidProbs.select(2, expertIdx);  // [batch, seq]
        Variable expertProbExpanded = expertProb.unsqueeze(2);     // [batch, seq, 1]

        // 计算 Top-K 归一化分母：每个位置被选中的 K 个专家的 sigmoid 概率之和
        // 使用 topKResult.indices 从 sigmoidProbs 中提取并求和
        float[][][] normFactors = new float[batchSize][seqLen][1];
        NdArray probsData = sigmoidProbs.getValue();
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                float sumProbs = 0.0f;
                for (int k = 0; k < config.getTopK(); k++) {
                    sumProbs += probsData.get(b, t, topKResult.indices[b][t][k]);
                }
                normFactors[b][t][0] = sumProbs > ZERO_THRESHOLD ? 1.0f / sumProbs : 1.0f / config.getTopK();
            }
        }

        // 选择掩码（不需要梯度）：标记该专家被选中的位置
        Variable selectionVar = new Variable(NdArray.of(selectionMask));
        selectionVar.setRequireGrad(false);

        // 归一化因子（不需要梯度）：每个位置的 Top-K 概率和的倒数
        Variable normVar = new Variable(NdArray.of(normFactors));
        normVar.setRequireGrad(false);

        // 权重 = expertProb * selectionMask * normFactor
        // expertProb 保持在计算图中，梯度可以回传到门控网络
        return expertProbExpanded.mul(selectionVar).mul(normVar);
    }

    /**
     * 专家网络内部类（SwiGLU FFN）
     * <p>
     * SwiGLU结构（对标DeepSeek-V3论文）：
     * output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
     * <p>
     * 相比GELU-FFN的改进：
     * - 门控机制过滤信息，训练更稳定
     * - SiLU激活函数比GELU计算简单且效果相当
     */
    private static class ExpertNetwork extends Module {
        private final Linear gateProj;   // 门控投影: inputDim -> hiddenDim
        private final SiLU silu;       // SiLU激活函数
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
        /**
         * MoE层的输出
         */
        public final Variable output;
        /**
         * 所有专家的门控概率
         */
        public final Variable gatingProbs;
        /**
         * Top-K选择结果
         */
        public final TopKResult topKResult;
        /**
         * 负载均衡损失
         */
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
