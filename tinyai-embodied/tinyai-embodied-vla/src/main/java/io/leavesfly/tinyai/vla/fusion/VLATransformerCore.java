package io.leavesfly.tinyai.vla.fusion;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * VLA Transformer核心
 * 多层Transformer实现多模态深度融合
 * 
 * @author TinyAI
 */
public class VLATransformerCore extends Module {
    
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
        super("VLATransformerCore");
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        
        // 初始化Transformer层
        this.layers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            layers.add(new VLATransformerLayer(hiddenDim, numHeads));
        }
        
        // 最终层归一化
        this.finalNorm = new LayerNorm("finalNorm", hiddenDim);
    }
    
    @Override
    public void resetParameters() {
        // 初始化已在构造函数中完成
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
    public Variable forward(Variable... inputs) {
        return fuse(inputs[0]);
    }
    
    /**
     * VLA Transformer层
     * 包含自注意力和前馈网络
     */
    private static class VLATransformerLayer extends Module {
        private final int hiddenDim;
        private final int numHeads;
        
        private CrossModalAttention selfAttention;
        private Linear ffn1;
        private Linear ffn2;
        private LayerNorm norm1;
        private LayerNorm norm2;
        
        public VLATransformerLayer(int hiddenDim, int numHeads) {
            super("VLATransformerLayer");
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            
            // 自注意力
            this.selfAttention = new CrossModalAttention(hiddenDim, numHeads);
            
            // 前馈网络
            this.ffn1 = new Linear("ffn1", hiddenDim, hiddenDim * 4, true);
            this.ffn2 = new Linear("ffn2", hiddenDim * 4, hiddenDim, true);
            
            // 层归一化
            this.norm1 = new LayerNorm("norm1", hiddenDim);
            this.norm2 = new LayerNorm("norm2", hiddenDim);
        }
        
        @Override
        public void resetParameters() {
            // 初始化已在构造函数中完成
        }
        
        @Override
        public Variable forward(Variable... inputs) {
            Variable input = inputs[0];
            // 自注意力 + 残差连接
            Variable normed1 = norm1.forward(input);
            Variable attnOut = selfAttention.forward(normed1);
            Variable residual1 = new Variable(
                input.getValue().add(attnOut.getValue())
            );
            
            // 前馈网络 + 残差连接
            Variable normed2 = norm2.forward(residual1);
            Variable ffn1Out = ffn1.forward(normed2);
            Variable geluOut = gelu(ffn1Out);
            Variable ffn2Out = ffn2.forward(geluOut);
            Variable residual2 = new Variable(
                residual1.getValue().add(ffn2Out.getValue())
            );
            
            return residual2;
        }
        
        /**
         * GELU激活函数
         */
        private Variable gelu(Variable input) {
            NdArray data = input.getValue();
            float[][] matrix = data.getMatrix();
            float[][] result = new float[matrix.length][matrix[0].length];
            
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    float x = matrix[i][j];
                    result[i][j] = (float)(0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * 
                               (x + 0.044715 * Math.pow(x, 3)))));
                }
            }
            
            return new Variable(NdArray.of(result));
        }
    }
}
