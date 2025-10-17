package io.leavesfly.tinyai.agent.vla.fusion;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.block.Block;
import io.leavesfly.tinyai.nnet.layer.Linear;
import io.leavesfly.tinyai.nnet.layer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * VLA Transformer核心
 * 多层Transformer实现多模态深度融合
 * 
 * @author TinyAI
 */
public class VLATransformerCore extends Block {
    
    private final int hiddenDim;
    private final int numLayers;
    private final int numHeads;
    
    private List<VLATransformerLayer> layers;
    private LayerNorm finalNorm;
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏维度
     * @param numLayers Transformer层数
     * @param numHeads 注意力头数
     */
    public VLATransformerCore(int hiddenDim, int numLayers, int numHeads) {
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        
        // 初始化Transformer层
        this.layers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            layers.add(new VLATransformerLayer(hiddenDim, numHeads));
        }
        
        // 最终层归一化
        this.finalNorm = new LayerNorm(hiddenDim);
    }
    
    /**
     * 融合多模态特征
     * 
     * @param input 拼接的多模态Token序列 [total_seq_len, hiddenDim]
     * @return 融合后的特征表示
     */
    public Variable fuse(Variable input) {
        Variable hidden = input;
        
        // 通过所有Transformer层
        for (VLATransformerLayer layer : layers) {
            hidden = layer.forward(hidden);
        }
        
        // 最终归一化
        Variable output = finalNorm.forward(hidden);
        
        return output;
    }
    
    @Override
    public Variable forward(Variable input) {
        return fuse(input);
    }
    
    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        for (VLATransformerLayer layer : layers) {
            params.addAll(layer.parameters());
        }
        params.addAll(finalNorm.parameters());
        return params;
    }
    
    /**
     * VLA Transformer层
     * 包含自注意力和前馈网络
     */
    private static class VLATransformerLayer extends Block {
        private final int hiddenDim;
        private final int numHeads;
        
        private CrossModalAttention selfAttention;
        private Linear ffn1;
        private Linear ffn2;
        private LayerNorm norm1;
        private LayerNorm norm2;
        
        public VLATransformerLayer(int hiddenDim, int numHeads) {
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            
            // 自注意力
            this.selfAttention = new CrossModalAttention(hiddenDim, numHeads);
            
            // 前馈网络
            this.ffn1 = new Linear(hiddenDim, hiddenDim * 4, true);
            this.ffn2 = new Linear(hiddenDim * 4, hiddenDim, true);
            
            // 层归一化
            this.norm1 = new LayerNorm(hiddenDim);
            this.norm2 = new LayerNorm(hiddenDim);
        }
        
        @Override
        public Variable forward(Variable input) {
            // 自注意力 + 残差连接
            Variable normed1 = norm1.forward(input);
            Variable attnOut = selfAttention.forward(normed1);
            Variable residual1 = new Variable(
                input.getData().add(attnOut.getData()), 
                input.requiresGrad()
            );
            
            // 前馈网络 + 残差连接
            Variable normed2 = norm2.forward(residual1);
            Variable ffn1Out = ffn1.forward(normed2);
            Variable geluOut = gelu(ffn1Out);
            Variable ffn2Out = ffn2.forward(geluOut);
            Variable residual2 = new Variable(
                residual1.getData().add(ffn2Out.getData()),
                input.requiresGrad()
            );
            
            return residual2;
        }
        
        /**
         * GELU激活函数
         */
        private Variable gelu(Variable input) {
            NdArray data = input.getData();
            double[] values = data.toDoubleArray();
            double[] result = new double[values.length];
            
            for (int i = 0; i < values.length; i++) {
                double x = values[i];
                result[i] = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * 
                           (x + 0.044715 * Math.pow(x, 3))));
            }
            
            return new Variable(new NdArray(result).reshape(data.getShape()), input.requiresGrad());
        }
        
        @Override
        public List<Variable> parameters() {
            List<Variable> params = new ArrayList<>();
            params.addAll(selfAttention.parameters());
            params.addAll(ffn1.parameters());
            params.addAll(ffn2.parameters());
            params.addAll(norm1.parameters());
            params.addAll(norm2.parameters());
            return params;
        }
    }
}
