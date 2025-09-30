package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;

/**
 * DeepSeek门控网络实现
 * 
 * 负责为DeepSeekMoE层选择Top-K专家并分配权重。
 * 包含噪声注入机制以改善负载均衡。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekGateNetwork extends Layer {
    
    private LinearLayer gateLinear;        // 门控线性层
    
    private int dModel;                    // 输入维度
    private int numExperts;                // 专家数量
    private int topK;                      // 选择的Top-K专家数量
    private boolean useNoise;              // 是否使用噪声
    private double noiseEpsilon;           // 噪声强度
    
    /**
     * 构造门控网络
     */
    public DeepSeekGateNetwork(String name, int dModel, int numExperts, int topK, 
                              boolean useNoise, double noiseEpsilon) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, numExperts));
        
        if (dModel <= 0 || numExperts <= 0 || topK <= 0) {
            throw new IllegalArgumentException("所有参数必须大于0");
        }
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK不能超过专家总数");
        }
        
        this.dModel = dModel;
        this.numExperts = numExperts;
        this.topK = topK;
        this.useNoise = useNoise;
        this.noiseEpsilon = noiseEpsilon;
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            gateLinear = new LinearLayer(
                name + "_gate_linear", 
                dModel, 
                numExperts, 
                false  // 不使用偏置
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
            gateLogits = addNoise(gateLogits);
        }
        
        // 重塑回三维并应用softmax
        NdArray logitsReshaped = gateLogits.getValue().reshape(
            Shape.of(batchSize, seqLen, numExperts)
        );
        
        // 逐位置应用softmax
        NdArray softmaxResult = applySoftmaxPerPosition(logitsReshaped, batchSize, seqLen);
        
        return new Variable(softmaxResult);
    }
    
    /**
     * 添加噪声以改善负载均衡
     */
    private Variable addNoise(Variable gateLogits) {
        if (noiseEpsilon <= 0.0) {
            return gateLogits;
        }
        
        NdArray logits = gateLogits.getValue();
        Shape shape = logits.getShape();
        
        // 创建高斯噪声
        NdArray noise = NdArray.likeRandomN(shape, 0);
        noise = noise.mulNum(noiseEpsilon);
        
        // 添加噪声
        return new Variable(logits.add(noise));
    }
    
    /**
     * 逐位置应用softmax
     */
    private NdArray applySoftmaxPerPosition(NdArray logits, int batchSize, int seqLen) {
        NdArray result = NdArray.zeros(logits.getShape());
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                // 获取当前位置的logits
                float[] positionLogits = new float[numExperts];
                for (int e = 0; e < numExperts; e++) {
                    positionLogits[e] = logits.get(b, s, e);
                }
                
                // 应用softmax
                float[] probabilities = applySoftmax(positionLogits);
                
                // 写回结果
                for (int e = 0; e < numExperts; e++) {
                    result.set(probabilities[e], b, s, e);
                }
            }
        }
        
        return result;
    }
    
    /**
     * 应用softmax函数
     */
    private float[] applySoftmax(float[] logits) {
        // 找到最大值（数值稳定性）
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            maxLogit = Math.max(maxLogit, logit);
        }
        
        // 计算exp和sum
        float[] exp = new float[logits.length];
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += exp[i];
        }
        
        // 归一化
        float[] probabilities = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exp[i] / sum;
        }
        
        return probabilities;
    }
    
    /**
     * 选择Top-K专家并计算相应权重
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
        
        // 简化的Top-K选择算法
        for (int k = 0; k < topK; k++) {
            int maxIdx = 0;
            float maxProb = expertProbs[0];
            
            // 找到最大概率的专家
            for (int i = 1; i < numExperts; i++) {
                if (expertProbs[i] > maxProb) {
                    maxProb = expertProbs[i];
                    maxIdx = i;
                }
            }
            
            // 记录Top-K专家
            indices[k] = maxIdx;
            weights[k] = maxProb;
            
            // 将已选专家的概率设为负数，避免重复选择
            expertProbs[maxIdx] = -1.0f;
        }
        
        // 重新归一化权重
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
    public int requireInputNum() {
        return 1;
    }
    
    // Getter方法
    public int getDModel() { return dModel; }
    public int getNumExperts() { return numExperts; }
    public int getTopK() { return topK; }
    public boolean isUseNoise() { return useNoise; }
    public double getNoiseEpsilon() { return noiseEpsilon; }
    
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
}