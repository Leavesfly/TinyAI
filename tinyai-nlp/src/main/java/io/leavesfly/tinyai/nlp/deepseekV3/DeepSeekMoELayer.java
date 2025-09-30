package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeekMoE层实现
 * 
 * DeepSeekMoE是DeepSeek-V3的核心组件，结合了标准MoE和共享专家的设计。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekMoELayer extends Layer {
    
    private int dModel;
    private int numExperts;
    private int dExpert;
    private int topK;
    private int numSharedExperts;
    
    private List<DeepSeekExpert> routedExperts;
    private List<DeepSeekExpert> sharedExperts;
    private DeepSeekGateNetwork gateNetwork;
    
    private boolean useSharedExperts;
    private double loadBalanceWeight;
    private double expertDropoutRate;
    
    private long totalTokens;
    private long[] expertUsageCount;
    
    public DeepSeekMoELayer(String name, int dModel, int numExperts, int dExpert, 
                           int topK, int numSharedExperts, double loadBalanceWeight, 
                           double expertDropoutRate) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        if (dModel <= 0 || numExperts <= 0 || dExpert <= 0 || topK <= 0) {
            throw new IllegalArgumentException("All dimension parameters must be positive");
        }
        if (topK > numExperts) {
            throw new IllegalArgumentException("topK cannot exceed number of experts");
        }
        
        this.dModel = dModel;
        this.numExperts = numExperts;
        this.dExpert = dExpert;
        this.topK = topK;
        this.numSharedExperts = numSharedExperts;
        this.loadBalanceWeight = loadBalanceWeight;
        this.expertDropoutRate = expertDropoutRate;
        
        this.useSharedExperts = numSharedExperts > 0;
        this.totalTokens = 0;
        this.expertUsageCount = new long[numExperts];
        
        init();
    }
    
    public DeepSeekMoELayer(String name, int dModel, int numExperts, int dExpert, int topK) {
        this(name, dModel, numExperts, dExpert, topK, 2, 0.001, 0.1);
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            // Initialize routed experts
            routedExperts = new ArrayList<DeepSeekExpert>();
            for (int i = 0; i < numExperts; i++) {
                DeepSeekExpert expert = new DeepSeekExpert(
                    name + "_routed_expert_" + i,
                    i,
                    dModel,
                    dExpert,
                    "SwiGLU"
                );
                routedExperts.add(expert);
            }
            
            // Initialize shared experts if enabled
            if (useSharedExperts) {
                sharedExperts = new ArrayList<DeepSeekExpert>();
                for (int i = 0; i < numSharedExperts; i++) {
                    DeepSeekExpert sharedExpert = new DeepSeekExpert(
                        name + "_shared_expert_" + i,
                        numExperts + i,
                        dModel,
                        dExpert,
                        "SwiGLU"
                    );
                    sharedExperts.add(sharedExpert);
                }
            }
            
            // Initialize gate network
            gateNetwork = new DeepSeekGateNetwork(
                name + "_gate",
                dModel,
                numExperts,
                topK,
                true,
                0.1
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
        
        if (inputData.getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        // Gate network selects experts
        DeepSeekGateNetwork.GateOutput gateOutput = gateNetwork.selectTopKExperts(input);
        
        // Compute routed expert output
        NdArray routedOutput = computeRoutedExpertOutput(input, gateOutput, batchSize, seqLen);
        
        // Compute shared expert output if enabled
        NdArray sharedOutput = null;
        if (useSharedExperts) {
            sharedOutput = computeSharedExpertOutput(input, batchSize, seqLen);
        }
        
        // Combine outputs
        NdArray finalOutput = combineExpertOutputs(routedOutput, sharedOutput, batchSize, seqLen);
        
        // Update statistics
        updateStatistics(gateOutput, batchSize, seqLen);
        
        return new Variable(finalOutput);
    }
    
    private NdArray computeRoutedExpertOutput(Variable input, 
                                             DeepSeekGateNetwork.GateOutput gateOutput,
                                             int batchSize, int seqLen) {
        NdArray output = NdArray.zeros(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                NdArray tokenInput = extractTokenInput(input.getValue(), b, s);
                Variable tokenVar = new Variable(tokenInput);
                
                for (int k = 0; k < topK; k++) {
                    int expertIdx = gateOutput.expertIndices[b][s][k];
                    float weight = gateOutput.expertWeights[b][s][k];
                    
                    if (expertIdx >= 0 && expertIdx < numExperts && weight > 0) {
                        Variable expertOutput = routedExperts.get(expertIdx).layerForward(tokenVar);
                        addWeightedOutput(output, expertOutput.getValue(), weight, b, s);
                    }
                }
            }
        }
        
        return output;
    }
    
    private NdArray computeSharedExpertOutput(Variable input, int batchSize, int seqLen) {
        NdArray output = NdArray.zeros(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                NdArray tokenInput = extractTokenInput(input.getValue(), b, s);
                Variable tokenVar = new Variable(tokenInput);
                
                for (DeepSeekExpert sharedExpert : sharedExperts) {
                    Variable expertOutput = sharedExpert.layerForward(tokenVar);
                    float weight = 1.0f / numSharedExperts;
                    addWeightedOutput(output, expertOutput.getValue(), weight, b, s);
                }
            }
        }
        
        return output;
    }
    
    private NdArray combineExpertOutputs(NdArray routedOutput, NdArray sharedOutput, 
                                        int batchSize, int seqLen) {
        if (!useSharedExperts || sharedOutput == null) {
            return routedOutput;
        }
        
        NdArray combinedOutput = NdArray.zeros(Shape.of(batchSize, seqLen, dModel));
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int d = 0; d < dModel; d++) {
                    float routedValue = routedOutput.get(b, s, d);
                    float sharedValue = sharedOutput.get(b, s, d);
                    float combinedValue = 0.7f * routedValue + 0.3f * sharedValue;
                    combinedOutput.set(combinedValue, b, s, d);
                }
            }
        }
        
        return combinedOutput;
    }
    
    private NdArray extractTokenInput(NdArray input, int batchIdx, int seqIdx) {
        NdArray tokenInput = NdArray.zeros(Shape.of(1, 1, dModel));
        
        for (int d = 0; d < dModel; d++) {
            float value = input.get(batchIdx, seqIdx, d);
            tokenInput.set(value, 0, 0, d);
        }
        
        return tokenInput;
    }
    
    private void addWeightedOutput(NdArray output, NdArray expertOutput, 
                                  float weight, int batchIdx, int seqIdx) {
        for (int d = 0; d < dModel; d++) {
            float expertValue = expertOutput.get(0, 0, d);
            float currentValue = output.get(batchIdx, seqIdx, d);
            float newValue = currentValue + weight * expertValue;
            output.set(newValue, batchIdx, seqIdx, d);
        }
    }
    
    private void updateStatistics(DeepSeekGateNetwork.GateOutput gateOutput, 
                                 int batchSize, int seqLen) {
        totalTokens += (long) batchSize * seqLen;
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int k = 0; k < topK; k++) {
                    int expertIdx = gateOutput.expertIndices[b][s][k];
                    if (expertIdx >= 0 && expertIdx < numExperts) {
                        expertUsageCount[expertIdx]++;
                    }
                }
            }
        }
    }
    
    public double computeLoadBalancingLoss() {
        if (totalTokens == 0) {
            return 0.0;
        }
        
        double averageUsage = 1.0 / numExperts;
        double variance = 0.0;
        
        for (int i = 0; i < numExperts; i++) {
            double usage = (double) expertUsageCount[i] / totalTokens;
            double diff = usage - averageUsage;
            variance += diff * diff;
        }
        
        return loadBalanceWeight * variance;
    }
    
    public void resetStats() {
        totalTokens = 0;
        for (int i = 0; i < numExperts; i++) {
            expertUsageCount[i] = 0;
        }
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // Getters
    public int getDModel() { return dModel; }
    public int getNumExperts() { return numExperts; }
    public int getDExpert() { return dExpert; }
    public int getTopK() { return topK; }
    public int getNumSharedExperts() { return numSharedExperts; }
    public long getTotalTokens() { return totalTokens; }
    public long[] getExpertUsageCount() { return expertUsageCount.clone(); }
    
    public List<DeepSeekExpert> getRoutedExperts() { return routedExperts; }
    public List<DeepSeekExpert> getSharedExperts() { return sharedExperts; }
    public DeepSeekGateNetwork getGateNetwork() { return gateNetwork; }
}