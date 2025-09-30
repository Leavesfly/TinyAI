package io.leavesfly.tinyai.nlp.deepseekV3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

/**
 * DeepSeek-V3 Transformer Block
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3TransformerBlock extends Layer {
    
    private int dModel;
    private int numHeads;
    private int dMLA;
    private int numExperts;
    private int dExpert;
    private int topK;
    private int numSharedExperts;
    private double dropoutRate;
    
    private LayerNorm inputLayerNorm;
    private MultiHeadLatentAttention mlaAttention;
    private LayerNorm moeLayerNorm;
    private DeepSeekMoELayer moeLayer;
    
    private boolean usePreNorm;
    
    public DeepSeekV3TransformerBlock(String name, DeepSeekV3Config config) {
        super(name, Shape.of(-1, -1, config.getDModel()), Shape.of(-1, -1, config.getDModel()));
        
        this.dModel = config.getDModel();
        this.numHeads = config.getNumHeads();
        this.dMLA = config.getDMLA();
        this.numExperts = config.getNumExperts();
        this.dExpert = config.getDExpert();
        this.topK = config.getTopK();
        this.numSharedExperts = config.getNumSharedExperts();
        this.dropoutRate = config.getDropoutRate();
        this.usePreNorm = true;
        
        init();
    }
    
    public DeepSeekV3TransformerBlock(String name, int dModel, int numHeads, 
                                     int dMLA, int numExperts, int dExpert, 
                                     int topK, double dropoutRate) {
        super(name, Shape.of(-1, -1, dModel), Shape.of(-1, -1, dModel));
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dMLA = dMLA;
        this.numExperts = numExperts;
        this.dExpert = dExpert;
        this.topK = topK;
        this.numSharedExperts = 2;
        this.dropoutRate = dropoutRate;
        this.usePreNorm = true;
        
        init();
    }
    
    @Override
    public void init() {
        if (!alreadyInit) {
            inputLayerNorm = new LayerNorm(name + "_input_norm", dModel);
            
            mlaAttention = new MultiHeadLatentAttention(
                name + "_mla_attention",
                dModel,
                numHeads,
                dMLA,
                Math.max(32, dMLA / 2),
                dropoutRate,
                true
            );
            
            moeLayerNorm = new LayerNorm(name + "_moe_norm", dModel);
            
            moeLayer = new DeepSeekMoELayer(
                name + "_moe_layer",
                dModel,
                numExperts,
                dExpert,
                topK,
                numSharedExperts,
                0.001,
                0.1
            );
            
            alreadyInit = true;
        }
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        
        if (input.getValue().getShape().getDimension(2) != dModel) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        if (usePreNorm) {
            return forwardPreNorm(input);
        } else {
            return forwardPostNorm(input);
        }
    }
    
    private Variable forwardPreNorm(Variable input) {
        // Attention sublayer
        Variable normalizedInput = inputLayerNorm.layerForward(input);
        Variable attentionOutput = mlaAttention.layerForward(normalizedInput);
        Variable afterAttention = addResidual(input, attentionOutput);
        
        // MoE sublayer
        Variable normalizedAfterAttention = moeLayerNorm.layerForward(afterAttention);
        Variable moeOutput = moeLayer.layerForward(normalizedAfterAttention);
        Variable finalOutput = addResidual(afterAttention, moeOutput);
        
        return finalOutput;
    }
    
    private Variable forwardPostNorm(Variable input) {
        // Attention sublayer
        Variable attentionOutput = mlaAttention.layerForward(input);
        Variable afterAttention = addResidual(input, attentionOutput);
        Variable normalizedAfterAttention = inputLayerNorm.layerForward(afterAttention);
        
        // MoE sublayer
        Variable moeOutput = moeLayer.layerForward(normalizedAfterAttention);
        Variable afterMoE = addResidual(normalizedAfterAttention, moeOutput);
        Variable finalOutput = moeLayerNorm.layerForward(afterMoE);
        
        return finalOutput;
    }
    
    private Variable addResidual(Variable input, Variable output) {
        return new Variable(input.getValue().add(output.getValue()));
    }
    
    public double computeLoadBalancingLoss() {
        return moeLayer.computeLoadBalancingLoss();
    }
    
    public void resetMoEStats() {
        moeLayer.resetStats();
    }
    
    public long[] getExpertUsageCount() {
        return moeLayer.getExpertUsageCount();
    }
    
    public long getTotalTokens() {
        return moeLayer.getTotalTokens();
    }
    
    public long getTotalParameterCount() {
        long totalParams = 0;
        
        // MLA attention parameters
        totalParams += (long) dModel * dModel; // Query projection
        totalParams += 2L * dModel * dMLA;     // Key/Value latent projection
        int headDim = dModel / numHeads;
        totalParams += 2L * dMLA * numHeads * headDim; // Key/Value decompression
        totalParams += (long) dModel * dModel; // Output projection
        
        // MoE layer parameters
        totalParams += (long) numExperts * 3 * dModel * dExpert; // Routed experts
        totalParams += (long) numSharedExperts * 3 * dModel * dExpert; // Shared experts
        totalParams += (long) dModel * numExperts; // Gate network
        
        // Layer normalization parameters
        totalParams += 4L * dModel;
        
        return totalParams;
    }
    
    public long getActiveParameterCount() {
        long activeParams = 0;
        
        // MLA attention (all active)
        activeParams += (long) dModel * dModel;
        activeParams += 2L * dModel * dMLA;
        int headDim = dModel / numHeads;
        activeParams += 2L * dMLA * numHeads * headDim;
        activeParams += (long) dModel * dModel;
        
        // MoE layer (only active experts)
        activeParams += (long) topK * 3 * dModel * dExpert; // Routed experts
        activeParams += (long) numSharedExperts * 3 * dModel * dExpert; // Shared experts
        activeParams += (long) dModel * numExperts; // Gate network
        
        // Layer normalization
        activeParams += 4L * dModel;
        
        return activeParams;
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
    
    // Getters
    public int getDModel() { return dModel; }
    public int getNumHeads() { return numHeads; }
    public int getDMLA() { return dMLA; }
    public int getNumExperts() { return numExperts; }
    public int getDExpert() { return dExpert; }
    public int getTopK() { return topK; }
    public int getNumSharedExperts() { return numSharedExperts; }
    public double getDropoutRate() { return dropoutRate; }
    
    public LayerNorm getInputLayerNorm() { return inputLayerNorm; }
    public MultiHeadLatentAttention getMlaAttention() { return mlaAttention; }
    public LayerNorm getMoeLayerNorm() { return moeLayerNorm; }
    public DeepSeekMoELayer getMoeLayer() { return moeLayer; }
}