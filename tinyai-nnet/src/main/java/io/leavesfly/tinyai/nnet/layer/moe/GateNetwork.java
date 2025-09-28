package io.leavesfly.tinyai.nnet.layer.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * MoE门控网络类
 * <p>
 * 门控网络负责为每个输入token决定应该激活哪些专家以及激活的权重。
 * 这是Mixture of Experts (MoE)架构的核心组件之一。
 * <p>
 * 工作原理：
 * 1. 接收输入token的表示
 * 2. 通过线性变换计算每个专家的得分（logits）
 * 3. 应用Softmax得到专家选择的概率分布
 * 4. 选择Top-K个专家并计算相应权重
 * 5. 可选地添加噪声来增强负载均衡
 * <p>
 * 特点：
 * - 支持Top-K专家选择策略
 * - 支持负载均衡（load balancing）
 * - 支持可训练的噪声注入
 * - 动态路由决策
 *
 * @author leavesfly
 * @version 1.0
 */
public class GateNetwork extends Layer {
    
    private LinearLayer gateLinear;        // 门控线性层
    private int dModel;                    // 输入维度
    private int numExperts;                // 专家数量
    private int topK;                      // 选择的Top-K专家数量
    private boolean useNoise;              // 是否使用噪声
    private double noiseEpsilon;           // 噪声强度
    private boolean useLoadBalancing;      // 是否启用负载均衡
    
    /**
     * 构造门控网络
     * 
     * @param name 门控网络名称
     * @param dModel 输入维度
     * @param numExperts 专家数量
     * @param topK 选择的Top-K专家数量
     * @param useNoise 是否使用噪声
     * @param noiseEpsilon 噪声强度
     * @param useLoadBalancing 是否启用负载均衡
     */
    public GateNetwork(String name, int dModel, int numExperts, int topK, 
                      boolean useNoise, double noiseEpsilon, boolean useLoadBalancing) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, numExperts));
        
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException(
                String.format("topK (%d) 必须在 1 到 numExperts (%d) 之间", topK, numExperts)
            );
        }
        
        this.dModel = dModel;
        this.numExperts = numExperts;
        this.topK = topK;
        this.useNoise = useNoise;
        this.noiseEpsilon = noiseEpsilon;
        this.useLoadBalancing = useLoadBalancing;
        
        init();
    }
    
    /**
     * 使用默认参数的构造函数
     */
    public GateNetwork(String name, int dModel, int numExperts, int topK) {
        this(name, dModel, numExperts, topK, true, 0.1, true);
    }
    
    /**
     * 简化的构造函数（默认Top-2专家）
     */
    public GateNetwork(String name, int dModel, int numExperts) {
        this(name, dModel, numExperts, 2);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // 门控线性层：dModel -> numExperts
            gateLinear = new LinearLayer(
                name + "_gate_linear", 
                dModel, 
                numExperts, 
                false  // 门控层通常不使用偏置
            );
            
            alreadyInit = true;
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
                String.format("门控网络输入维度不匹配。期望%d，实际%d", 
                             dModel, inputData.getShape().getDimension(2))
            );
        }
        
        // 将三维输入重塑为二维进行线性变换
        NdArray inputReshaped = inputData.reshape(Shape.of(batchSize * seqLen, dModel));
        Variable reshapedInput = new Variable(inputReshaped);
        
        // 计算门控logits
        Variable gateLogits = gateLinear.layerForward(reshapedInput);
        
        // 添加噪声（如果启用）
        if (useNoise) {
            gateLogits = addGatingNoise(gateLogits);
        }
        
        // 计算Softmax概率
        Variable gateProbabilities = applySoftmax(gateLogits);
        
        // 重塑回三维: (batch_size, seq_len, num_experts)
        NdArray probsReshaped = gateProbabilities.getValue().reshape(
            Shape.of(batchSize, seqLen, numExperts)
        );
        
        return new Variable(probsReshaped);
    }
    
    /**
     * 添加门控噪声以改善负载均衡
     */
    private Variable addGatingNoise(Variable logits) {
        if (noiseEpsilon <= 0.0) {
            return logits;
        }
        
        // 生成随机噪声
        NdArray logitsData = logits.getValue();
        NdArray noise = NdArray.likeRandomN(logitsData.getShape()).mulNum((float) noiseEpsilon);
        
        return logits.add(new Variable(noise));
    }
    
    /**
     * 应用Softmax计算专家选择概率
     */
    private Variable applySoftmax(Variable logits) {
        NdArray logitsData = logits.getValue();
        NdArray softmaxResult = logitsData.softMax();
        return new Variable(softmaxResult);
    }
    
    /**
     * 选择Top-K专家并计算相应权重
     * 返回专家索引和对应的权重
     */
    public GateOutput selectTopKExperts(Variable input) {
        // 先计算门控概率
        Variable gateProbabilities = layerForward(input);
        NdArray probsData = gateProbabilities.getValue();
        
        int batchSize = probsData.getShape().getDimension(0);
        int seqLen = probsData.getShape().getDimension(1);
        
        // 存储每个位置的Top-K专家索引和权重
        int[][][] topKIndices = new int[batchSize][seqLen][topK];
        float[][][] topKWeights = new float[batchSize][seqLen][topK];
        
        // 为每个token选择Top-K专家
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                selectTopKForPosition(probsData, b, s, topKIndices[b][s], topKWeights[b][s]);
            }
        }
        
        return new GateOutput(topKIndices, topKWeights, gateProbabilities);
    }
    
    /**
     * 为特定位置选择Top-K专家
     */
    private void selectTopKForPosition(NdArray probs, int batchIdx, int seqIdx, 
                                     int[] indices, float[] weights) {
        // 获取当前位置的所有专家概率
        float[] expertProbs = new float[numExperts];
        for (int i = 0; i < numExperts; i++) {
            expertProbs[i] = probs.get(batchIdx, seqIdx, i);
        }
        
        // 简单的Top-K选择算法（可以优化为更高效的实现）
        for (int k = 0; k < topK; k++) {
            int maxIdx = -1;
            float maxProb = Float.NEGATIVE_INFINITY;
            
            // 找到剩余专家中概率最大的
            for (int i = 0; i < numExperts; i++) {
                if (expertProbs[i] > maxProb) {
                    boolean alreadySelected = false;
                    for (int j = 0; j < k; j++) {
                        if (indices[j] == i) {
                            alreadySelected = true;
                            break;
                        }
                    }
                    if (!alreadySelected) {
                        maxIdx = i;
                        maxProb = expertProbs[i];
                    }
                }
            }
            
            indices[k] = maxIdx;
            weights[k] = maxProb;
        }
        
        // 归一化权重
        float totalWeight = 0.0f;
        for (float weight : weights) {
            totalWeight += weight;
        }
        if (totalWeight > 0.0f) {
            for (int k = 0; k < topK; k++) {
                weights[k] /= totalWeight;
            }
        }
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        return layerForward(new Variable(inputs[0])).getValue();
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        List<NdArray> result = new ArrayList<>();
        result.add(yGrad);
        return result;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getNumExperts() { return numExperts; }
    public int getTopK() { return topK; }
    public boolean isUseNoise() { return useNoise; }
    public double getNoiseEpsilon() { return noiseEpsilon; }
    public boolean isUseLoadBalancing() { return useLoadBalancing; }
    public LinearLayer getGateLinear() { return gateLinear; }
    
    /**
     * 门控网络输出结果类
     */
    public static class GateOutput {
        public final int[][][] expertIndices;    // [batch_size, seq_len, topK]
        public final float[][][] expertWeights;  // [batch_size, seq_len, topK]
        public final Variable gateProbabilities; // 完整的专家概率分布
        
        public GateOutput(int[][][] expertIndices, float[][][] expertWeights, Variable gateProbabilities) {
            this.expertIndices = expertIndices;
            this.expertWeights = expertWeights;
            this.gateProbabilities = gateProbabilities;
        }
    }
    
    @Override
    public String toString() {
        return String.format("GateNetwork(dModel=%d, numExperts=%d, topK=%d, noise=%s, loadBalance=%s)", 
                           dModel, numExperts, topK, useNoise, useLoadBalancing);
    }
}