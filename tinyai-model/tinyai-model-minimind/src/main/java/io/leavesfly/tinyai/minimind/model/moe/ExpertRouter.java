package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * Expert Router - 专家路由网络
 * 
 * 根据输入计算每个专家的权重,并选择Top-K个专家激活
 * 
 * 核心功能:
 * 1. 计算门控权重: gate_logits = W_g · x
 * 2. Top-K选择: 只保留权重最大的K个专家
 * 3. Softmax归一化: 确保权重和为1
 * 4. Noisy Top-K: 添加噪声避免专家过载
 * 
 * 路由公式:
 * w_i = Softmax(W_g · x + noise)
 * Top-K: 选择权重最大的K个专家
 * 
 * @author leavesfly
 * @since 2024
 */
public class ExpertRouter extends Module {
    
    private final int inputDim;
    private final int numExperts;
    private final int topK;
    private final float noiseFactor;
    
    private final Linear gateLinear;  // 门控线性层: input_dim -> num_experts
    
    /**
     * 构造函数
     * 
     * @param inputDim 输入维度
     * @param numExperts 专家数量
     * @param topK Top-K选择数量
     * @param noiseFactor 噪声因子(用于负载均衡)
     */
    public ExpertRouter(int inputDim, int numExperts, int topK, float noiseFactor) {
        super("expert_router");
        
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK must be <= numExperts");
        }
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be >= 1");
        }
        
        this.inputDim = inputDim;
        this.numExperts = numExperts;
        this.topK = topK;
        this.noiseFactor = noiseFactor;
        
        // 创建门控层
        this.gateLinear = new Linear("gate", inputDim, numExperts, true);
        
        // 注册子模块
        registerModule("gate", gateLinear);
    }
    
    /**
     * 前向传播(Variable版本)
     * <p>
     * 委托到forwardRouter，返回保留计算图连通的topKWeightsVar。
     * 如需完整路由信息(indices等)，请直接调用forwardRouter。
     */
    @Override
    public Variable forward(Variable... inputs) {
        RouterOutput routerOutput = forwardRouter(inputs[0]);
        return routerOutput.getTopKWeightsVar();
    }
    
    /**
     * 前向传播(NdArray版本，仅用于推理)
     * <p>
     * 注意：此方法通过 new Variable 包装输入，不保持上游计算图连通。
     * 训练时应使用 forwardRouter(Variable) 方法。
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        input.setRequireGrad(false);
        return forward(input).getValue();
    }
    
    /**
     * 前向传播(返回RouterOutput)
     * 
     * @param input 输入 [batch_size, input_dim]
     * @return 路由结果 RouterOutput
     */
    public RouterOutput forwardRouter(Variable input) {
        // 支持3D输入 [batch_size, seq_len, input_dim] -> reshape为 [batch_size * seq_len, input_dim]
        Variable flatInput = input;
        if (input.getShape().getDimNum() == 3) {
            int batchSize = input.getShape().getDimension(0);
            int seqLen = input.getShape().getDimension(1);
            flatInput = input.reshape(Shape.of(batchSize * seqLen, inputDim));
        }
        int flatBatchSize = flatInput.getShape().getDimension(0);
        
        // 1. 计算门控logits: [flatBatchSize, num_experts]
        Variable gateLogits = gateLinear.forward(flatInput);
        
        // 2. 添加噪声(训练时)
        if (isTraining() && noiseFactor > 0) {
            gateLogits = addNoise(gateLogits);
        }
        
        // 3. Top-K选择和Softmax
        RouterOutput output = topKGating(gateLogits, flatBatchSize);
        
        return output;
    }
    
    /**
     * Top-K门控计算
     * <p>
     * 修复说明：构建携带完整计算图的 topKWeightsVar 和 allWeightsVar，
     * 使得 MoE 主路径的加权合并能够把梯度回传到 gateLinear。
     * 步骤：
     *   1) 对整个 batch 一次性做 softmax：allWeightsVar = softmax(gateLogits)  [B, E]
     *   2) 用 float[] 排序确定每个样本的 topKIndices（离散、不可导）
     *   3) 将全局 topKIndices 展平成一维 "全局列表索引"，一次性 indexSelect 从
     *      allWeightsVar.reshape([B*E]) 中切出 topK 权重，再 reshape 回 [B, topK]
     *   4) 在 Variable 层面做 topK 权重归一化（sum + div + broadcast）
     * 
     * @param gateLogits 门控logits [batch_size, num_experts]
     * @param batchSize 批次大小
     * @return 路由结果
     */
    private RouterOutput topKGating(Variable gateLogits, int batchSize) {
        // 1) 一次性对整个 batch 做 softmax，保持计算图连通
        //    allWeightsVar: [batchSize, numExperts]
        Variable allWeightsVar = softmaxLastDim(gateLogits, batchSize, numExperts);

        // 2) 从 NdArray 层面读取数据做离散的 Top-K 排序（这一步本就不可导）
        NdArray allWeightsData = allWeightsVar.getValue();
        float[] softmaxBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) allWeightsData).buffer;

        int[][] topKIndices = new int[batchSize][topK];
        float[][] allWeights = new float[batchSize][numExperts];
        // 展平的全局索引：gather 位置 = b * numExperts + expertIdx
        float[] flatGatherIndices = new float[batchSize * topK];

        for (int b = 0; b < batchSize; b++) {
            int rowOffset = b * numExperts;
            System.arraycopy(softmaxBuffer, rowOffset, allWeights[b], 0, numExperts);

            // 按权重排序（离散操作）
            Integer[] indices = new Integer[numExperts];
            for (int i = 0; i < numExperts; i++) {
                indices[i] = i;
            }
            final float[] rowRef = allWeights[b];
            Arrays.sort(indices, Comparator.comparingDouble(i -> -rowRef[i]));

            for (int k = 0; k < topK; k++) {
                topKIndices[b][k] = indices[k];
                flatGatherIndices[b * topK + k] = rowOffset + indices[k];
            }
        }

        // 3) 用 indexSelect 从 allWeightsVar(reshape 成 [B*E]) 中切出 topK 权重，保留计算图
        Variable flatWeightsVar = allWeightsVar.reshape(Shape.of(batchSize * numExperts));
        Variable gatherIdxVar = new Variable(
                NdArray.of(flatGatherIndices, Shape.of(batchSize * topK)));
        gatherIdxVar.setRequireGrad(false);
        Variable topKSelectedVar = flatWeightsVar.indexSelect(0, gatherIdxVar);
        // reshape 回 [B, topK]
        Variable topKWeightsVarRaw = topKSelectedVar.reshape(Shape.of(batchSize, topK));

        // 4) 在 Variable 层面重新归一化：topKWeightsVar = raw / sum(raw, dim=1, keepdim=true)
        //    sum 用 sumTo 到 [B, 1]，再 broadcastTo 回 [B, topK]
        Variable rowSumVar = topKWeightsVarRaw.sumTo(Shape.of(batchSize, 1));
        // 避免除零：加一个极小常量（不参与梯度）
        Variable epsVar = new Variable(NdArray.of(new float[]{1e-12f}, Shape.of(1)));
        epsVar.setRequireGrad(false);
        rowSumVar = rowSumVar.add(epsVar.broadcastTo(Shape.of(batchSize, 1)));
        Variable rowSumBroadcast = rowSumVar.broadcastTo(Shape.of(batchSize, topK));
        Variable topKWeightsVar = topKWeightsVarRaw.div(rowSumBroadcast);

        // 同步回 float[][] 形式（供统计/负载均衡兼容使用）
        float[][] topKWeightsFloat = new float[batchSize][topK];
        float[] normalizedBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) topKWeightsVar.getValue()).buffer;
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(normalizedBuffer, b * topK, topKWeightsFloat[b], 0, topK);
        }

        return new RouterOutput(topKIndices, topKWeightsFloat, allWeights,
                topKWeightsVar, allWeightsVar);
    }

    /**
     * 在最后一维做 softmax（[B, E] -> [B, E]），Reshape 仅保持计算图连通
     */
    private Variable softmaxLastDim(Variable input, int batchSize, int numExperts) {
        // 当前输入已是 [B, E]，softMax() 对整张量做全局 softmax 不符合要求，
        // 需要按行 softmax：利用已有实现 —— Variable.softMax() 是全局的，
        // 但在 2D 情况下，TinyAI 的 SoftMax 是沿最后一维（见 MultiHeadAttention 中用法）。
        // 这里与 MultiHeadAttention 的 softmaxLastDim 语义保持一致。
        return input.softMax();
    }
    
    /**
     * 添加噪声(Noisy Top-K Gating)
     * 
     * 使用标准高斯噪声,噪声幅度由 noiseFactor 控制:
     * noisy_logits = logits + StandardNormal() * noiseFactor
     * 
     * 参考: Shazeer et al. 2017 "Outrageously Large Neural Networks"
     * 
     * 修复说明: 通过 Variable 运算添加噪声,确保噪声参与梯度计算
     */
    private Variable addNoise(Variable gateLogits) {
        NdArray logitsData = gateLogits.getValue();
        int[] shape = logitsData.getShape().getShapeDims();
        int totalElements = 1;
        for (int dim : shape) {
            totalElements *= dim;
        }
        
        // 创建噪声数组
        float[] noiseBuffer = new float[totalElements];
        Random random = new Random();
        for (int i = 0; i < totalElements; i++) {
            noiseBuffer[i] = (float) random.nextGaussian() * noiseFactor;
        }
        
        // 将噪声包装为 Variable(不需要梯度)
        NdArray noiseNdArray = NdArray.of(noiseBuffer, logitsData.getShape());
        Variable noiseVar = new Variable(noiseNdArray);
        noiseVar.setRequireGrad(false);
        
        // 通过 Variable.add() 运算添加噪声,确保噪声通过计算图传播
        return gateLogits.add(noiseVar);
    }
    
    // 已删除 softmax 方法，改用 Variable.softMax() 算子
    
    /**
     * 获取输入维度
     */
    public int getInputDim() {
        return inputDim;
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
        return String.format("ExpertRouter(input=%d, experts=%d, topK=%d, noise=%.3f)",
            inputDim, numExperts, topK, noiseFactor);
    }
    
    /**
     * 路由输出结果
     * <p>
     * 包含两种形态的数据：
     *   - 离散快照（int[][] / float[][]）：用于统计、负载均衡的 CV 计算等无需梯度的场景
     *   - Variable 形态（topKWeightsVar / allWeightsVar）：保留完整计算图，
     *     供 MoE 主路径和负载均衡损失使用，确保 gate 参数可被正确更新
     */
    public static class RouterOutput {
        private final int[][] topKIndices;          // [batch_size, topK]
        private final float[][] topKWeights;        // [batch_size, topK]
        private final float[][] allWeights;         // [batch_size, num_experts]

        /** 归一化后的 topK 权重，保留计算图；形状 [batch_size, topK] */
        private final Variable topKWeightsVar;
        /** 所有专家的 softmax 权重，保留计算图；形状 [batch_size, num_experts] */
        private final Variable allWeightsVar;

        public RouterOutput(int[][] topKIndices, float[][] topKWeights, float[][] allWeights) {
            this(topKIndices, topKWeights, allWeights, null, null);
        }

        public RouterOutput(int[][] topKIndices, float[][] topKWeights, float[][] allWeights,
                            Variable topKWeightsVar, Variable allWeightsVar) {
            this.topKIndices = topKIndices;
            this.topKWeights = topKWeights;
            this.allWeights = allWeights;
            this.topKWeightsVar = topKWeightsVar;
            this.allWeightsVar = allWeightsVar;
        }

        public int[][] getTopKIndices() {
            return topKIndices;
        }

        public float[][] getTopKWeights() {
            return topKWeights;
        }

        public float[][] getAllWeights() {
            return allWeights;
        }

        public int getBatchSize() {
            return topKIndices.length;
        }

        public int getTopK() {
            return topKIndices[0].length;
        }

        /** 获取保留计算图的 topK 权重 Variable，可能为 null（旧调用方式） */
        public Variable getTopKWeightsVar() {
            return topKWeightsVar;
        }

        /** 获取保留计算图的所有专家权重 Variable，可能为 null（旧调用方式） */
        public Variable getAllWeightsVar() {
            return allWeightsVar;
        }
    }
}