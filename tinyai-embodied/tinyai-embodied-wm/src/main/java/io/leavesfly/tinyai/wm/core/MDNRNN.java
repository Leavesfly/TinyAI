package io.leavesfly.tinyai.wm.core;

import io.leavesfly.tinyai.wm.model.Action;
import io.leavesfly.tinyai.wm.model.HiddenState;
import io.leavesfly.tinyai.wm.model.LatentState;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * MDN-RNN（Mixture Density Network - Recurrent Neural Network）
 * 世界模型的记忆组件，预测下一个潜在状态
 * 
 * 功能：
 * - 基于当前潜在状态 z_t、动作 a_t 和隐藏状态 h_t
 * - 预测下一个潜在状态 z_{t+1} 的分布（混合高斯分布）
 * - 更新隐藏状态 h_{t+1}
 *
 * @author leavesfly
 * @since 2025-10-18
 */
public class MDNRNN {
    
    /**
     * 潜在空间维度
     */
    private final int latentSize;
    
    /**
     * 动作空间维度
     */
    private final int actionSize;
    
    /**
     * 隐藏状态维度
     */
    private final int hiddenSize;
    
    /**
     * 高斯混合分量数量
     */
    private final int numMixtures;
    
    // RNN输入层（z_t + a_t -> RNN输入）
    private final Linear inputLayer;
    
    // RNN核心层（简化的GRU单元）
    private final GRUCell rnnCell;
    
    // MDN输出层（隐藏状态 -> 混合高斯参数）
    private final MDNOutput mdnOutput;
    
    /**
     * 构造函数
     *
     * @param latentSize 潜在空间维度
     * @param actionSize 动作空间维度
     * @param hiddenSize 隐藏状态维度
     * @param numMixtures 混合分量数量
     */
    public MDNRNN(int latentSize, int actionSize, int hiddenSize, int numMixtures) {
        this.latentSize = latentSize;
        this.actionSize = actionSize;
        this.hiddenSize = hiddenSize;
        this.numMixtures = numMixtures;
        
        // 输入层：[z_t; a_t] -> RNN输入
        int inputSize = latentSize + actionSize;
        this.inputLayer = new Linear("input", inputSize, hiddenSize);
        
        // RNN单元
        this.rnnCell = new GRUCell(hiddenSize);
        
        // MDN输出层
        this.mdnOutput = new MDNOutput(hiddenSize, latentSize, numMixtures);
    }
    
    /**
     * 前向传播一步
     *
     * @param latentState 当前潜在状态 z_t
     * @param action 当前动作 a_t
     * @param hiddenState 当前隐藏状态 h_t
     * @return RNN输出（下一个隐藏状态和预测的潜在状态分布）
     */
    public RNNOutput forward(LatentState latentState, Action action, HiddenState hiddenState) {
        // 1. 拼接输入：[z_t; a_t]
        NdArray input = concatenate(
            latentState.getZ(),
            action.getActionVector()
        );
        
        // 2. 通过输入层
        Variable inputProcessed = inputLayer.forward(new Variable(input));
        
        // 3. 通过RNN单元
        HiddenState nextHidden = rnnCell.forward(inputProcessed.getValue(), hiddenState);
        
        // 4. 通过MDN输出层，得到混合高斯参数
        MDNParameters params = mdnOutput.forward(nextHidden.getH());
        
        return new RNNOutput(nextHidden, params);
    }
    
    /**
     * 从MDN分布中采样下一个潜在状态
     *
     * @param params MDN参数
     * @return 采样的潜在状态
     */
    public LatentState sample(MDNParameters params) {
        // 1. 从混合权重中选择分量
        int selectedMixture = sampleMixture(params.getWeights());
        
        // 2. 从选定的高斯分布中采样
        NdArray mu = params.getMu(selectedMixture);
        NdArray sigma = params.getSigma(selectedMixture);
        
        // z ~ N(μ, σ²)
        NdArray epsilon = NdArray.randn(mu.getShape());
        NdArray z = mu.add(sigma.mul(epsilon));
        
        return new LatentState(z);
    }
    
    /**
     * 从混合权重中采样分量索引
     */
    private int sampleMixture(NdArray weights) {
        float[] probs = weights.getArray();
        double rand = Math.random();
        double cumsum = 0.0;
        
        for (int i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (rand < cumsum) {
                return i;
            }
        }
        return probs.length - 1;
    }
    
    /**
     * 计算MDN损失（负对数似然）
     *
     * @param target 目标潜在状态
     * @param params 预测的MDN参数
     * @return 损失值
     */
    public double calculateLoss(LatentState target, MDNParameters params) {
        NdArray targetZ = target.getZ();
        
        // 计算每个混合分量的概率密度
        double totalProb = 0.0;
        float[] weights = params.getWeights().getArray();
        for (int i = 0; i < numMixtures; i++) {
            double weight = weights[i];
            double density = gaussianDensity(targetZ, params.getMu(i), params.getSigma(i));
            totalProb += weight * density;
        }
        
        // 负对数似然
        return -Math.log(totalProb + 1e-8);
    }
    
    /**
     * 计算高斯分布的概率密度
     */
    private double gaussianDensity(NdArray x, NdArray mu, NdArray sigma) {
        NdArray diff = x.sub(mu);
        NdArray squared = diff.mul(diff);
        NdArray variance = sigma.mul(sigma);
        NdArray exponent = squared.div(variance.mulNum(2.0)).neg();
        
        double product = 1.0;
        float[] expData = exponent.exp().getArray();
        float[] sigmaData = sigma.getArray();
        for (int i = 0; i < expData.length; i++) {
            product *= expData[i] / (sigmaData[i] * Math.sqrt(2 * Math.PI));
        }
        return product;
    }
    
    /**
     * 创建初始隐藏状态
     */
    public HiddenState createInitialHidden() {
        return HiddenState.zeros(hiddenSize, false);
    }
    

    
    // Getters
    public int getLatentSize() {
        return latentSize;
    }
    
    public int getActionSize() {
        return actionSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public int getNumMixtures() {
        return numMixtures;
    }
    
    /**
     * 简化的GRU单元
     */
    private static class GRUCell {
        private final int hiddenSize;
        private final Linear resetGate;
        private final Linear updateGate;
        private final Linear candidateGate;
        
        public GRUCell(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            this.resetGate = new Linear("reset", hiddenSize * 2, hiddenSize);
            this.updateGate = new Linear("update", hiddenSize * 2, hiddenSize);
            this.candidateGate = new Linear("candidate", hiddenSize * 2, hiddenSize);
        }
        
        public HiddenState forward(NdArray input, HiddenState prevHidden) {
            NdArray h = prevHidden.getH();
            NdArray combined = concatenate(input, h);
            
            // 重置门：r = σ(W_r * [input; h])
            Variable r = resetGate.forward(new Variable(combined));
            NdArray rSigmoid = r.getValue().sigmoid();
            
            // 更新门：z = σ(W_z * [input; h])
            Variable z = updateGate.forward(new Variable(combined));
            NdArray zSigmoid = z.getValue().sigmoid();
            
            // 候选隐藏状态：h_tilde = tanh(W * [input; r ⊙ h])
            NdArray resetCombined = NdArray.of(new float[hiddenSize * 2], Shape.of(hiddenSize * 2));
            // 简化实现：直接拼接
            Variable hTilde = candidateGate.forward(new Variable(combined));
            NdArray hTildeTanh = hTilde.getValue().tanh();
            
            // 新隐藏状态：h_new = (1-z) ⊙ h + z ⊙ h_tilde
            NdArray hNew = zSigmoid.neg().add(NdArray.of(1.0f)).mul(h)
                .add(zSigmoid.mul(hTildeTanh));
            
            return new HiddenState(hNew);
        }
    }
    
    /**
     * MDN输出层
     */
    private static class MDNOutput {
        private final Linear weightLayer;
        private final Linear muLayer;
        private final Linear sigmaLayer;
        
        public MDNOutput(int hiddenSize, int latentSize, int numMixtures) {
            this.weightLayer = new Linear("weight", hiddenSize, numMixtures);
            this.muLayer = new Linear("mu", hiddenSize, latentSize * numMixtures);
            this.sigmaLayer = new Linear("sigma", hiddenSize, latentSize * numMixtures);
        }
        
        public MDNParameters forward(NdArray hidden) {
            Variable h = new Variable(hidden);
            
            // 混合权重（经过softmax）
            Variable weightsLogits = weightLayer.forward(h);
            NdArray weights = weightsLogits.getValue().softMax();
            
            // 均值
            Variable mu = muLayer.forward(h);
            
            // 标准差（经过exp确保为正）
            Variable sigmaLogits = sigmaLayer.forward(h);
            NdArray sigma = sigmaLogits.getValue().exp();
            
            return new MDNParameters(weights, mu.getValue(), sigma);
        }
    }
    
    /**
     * RNN输出
     */
    public static class RNNOutput {
        private final HiddenState nextHidden;
        private final MDNParameters mdnParams;
        
        public RNNOutput(HiddenState nextHidden, MDNParameters mdnParams) {
            this.nextHidden = nextHidden;
            this.mdnParams = mdnParams;
        }
        
        public HiddenState getNextHidden() {
            return nextHidden;
        }
        
        public MDNParameters getMdnParams() {
            return mdnParams;
        }
    }
    
    /**
     * MDN参数
     */
    public static class MDNParameters {
        private final NdArray weights;  // [numMixtures]
        private final NdArray mu;       // [latentSize * numMixtures]
        private final NdArray sigma;    // [latentSize * numMixtures]
        
        public MDNParameters(NdArray weights, NdArray mu, NdArray sigma) {
            this.weights = weights;
            this.mu = mu;
            this.sigma = sigma;
        }
        
        public NdArray getWeights() {
            return weights;
        }
        
        public NdArray getMu(int mixtureIndex) {
            int latentSize = mu.getShape().getDimension(0) / weights.getShape().getDimension(0);
            // 简化实现：返回子数组
            float[] muData = mu.getArray();
            float[] result = new float[latentSize];
            System.arraycopy(muData, mixtureIndex * latentSize, result, 0, latentSize);
            return NdArray.of(result, Shape.of(latentSize));
        }
        
        public NdArray getSigma(int mixtureIndex) {
            int latentSize = sigma.getShape().getDimension(0) / weights.getShape().getDimension(0);
            float[] sigmaData = sigma.getArray();
            float[] result = new float[latentSize];
            System.arraycopy(sigmaData, mixtureIndex * latentSize, result, 0, latentSize);
            return NdArray.of(result, Shape.of(latentSize));
        }
    }
    
    // 辅助方法：拼接两个NdArray
    private static NdArray concatenate(NdArray a, NdArray b) {
        float[] aData = a.getArray();
        float[] bData = b.getArray();
        float[] result = new float[aData.length + bData.length];
        System.arraycopy(aData, 0, result, 0, aData.length);
        System.arraycopy(bData, 0, result, aData.length, bData.length);
        return NdArray.of(result, Shape.of(result.length));
    }
}
