package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.LanguageInput;
import io.leavesfly.tinyai.agent.vla.utils.Tokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.Block;
import io.leavesfly.tinyai.nnet.layer.dnn.LinearLayer;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.transformer.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * 语言编码器
 * 基于Transformer编码自然语言指令
 * 复用GPT模型的Transformer结构
 * 
 * @author TinyAI
 */
public class LanguageEncoder extends Block {
    
    private final int vocabSize;
    private final int hiddenDim;
    private final int maxSeqLen;
    private final int numLayers;
    
    // Token嵌入层
    private Embedding tokenEmbedding;
    
    // 位置嵌入（学习式）
    private NdArray positionalEmbedding;
    
    // Transformer层（简化实现）
    private List<TransformerEncoderLayer> transformerLayers;
    
    // 层归一化
    private LayerNorm layerNorm;
    
    // 分词器
    private Tokenizer tokenizer;
    
    /**
     * 构造函数
     * 
     * @param vocabSize 词汇表大小
     * @param hiddenDim 隐藏层维度
     * @param maxSeqLen 最大序列长度
     * @param numLayers Transformer层数
     */
    public LanguageEncoder(int vocabSize, int hiddenDim, int maxSeqLen, int numLayers) {
        super("LanguageEncoder", null);
        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        
        // 初始化Token嵌入
        this.tokenEmbedding = new Embedding("tokenEmbed", vocabSize, hiddenDim);
        
        // 初始化位置嵌入（可学习）
        this.positionalEmbedding = createLearnablePositionalEmbedding(maxSeqLen, hiddenDim);
        
        // 初始化Transformer层
        this.transformerLayers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            transformerLayers.add(new TransformerEncoderLayer(hiddenDim, 8));
        }
        
        // 层归一化
        this.layerNorm = new LayerNorm("layerNorm", hiddenDim);
        
        // 初始化分词器
        this.tokenizer = new Tokenizer();
    }
    
    @Override
    public void init() {
        // 初始化已在构造函数中完成
    }
    
    /**
     * 编码语言输入
     * 
     * @param languageInput 语言输入
     * @return 语言Token序列，维度 [seq_len, hiddenDim]
     */
    public NdArray encode(LanguageInput languageInput) {
        String instruction = languageInput.getInstruction();
        
        // 分词
        int[] tokenIds = tokenizer.encode(instruction);
        int seqLen = Math.min(tokenIds.length, maxSeqLen);
        
        // 创建token IDs的NdArray
        double[] tokenIdsDouble = new double[seqLen];
        for (int i = 0; i < seqLen; i++) {
            tokenIdsDouble[i] = tokenIds[i];
        }
        NdArray tokenIdsArray = NdArray.of(tokenIdsDouble).reshape(io.leavesfly.tinyai.ndarr.Shape.of(seqLen, 1));
        
        // Token嵌入
        Variable tokenEmbedVar = tokenEmbedding.layerForward(new Variable(tokenIdsArray));
        NdArray tokenEmbed = tokenEmbedVar.getValue();
        
        // 添加位置嵌入
        NdArray posEmbed = sliceArray(positionalEmbedding, 0, seqLen, 0, hiddenDim);
        NdArray embeddings = tokenEmbed.add(posEmbed);
        
        // 通过Transformer层
        Variable hidden = new Variable(embeddings);
        for (TransformerEncoderLayer layer : transformerLayers) {
            hidden = layer.layerForward(hidden);
        }
        
        // 最终层归一化
        Variable output = layerNorm.layerForward(hidden);
        
        // 保存到languageInput
        languageInput.setTokenIds(tokenIdsArray);
        languageInput.setEmbeddings(output.getValue());
        
        return output.getValue();
    }
    
    private NdArray sliceArray(NdArray array, int startRow, int endRow, int startCol, int endCol) {
        float[][] matrix = array.getMatrix();
        float[][] result = new float[endRow - startRow][endCol - startCol];
        
        for (int i = 0; i < endRow - startRow; i++) {
            for (int j = 0; j < endCol - startCol; j++) {
                result[i][j] = matrix[i + startRow][j + startCol];
            }
        }
        
        return NdArray.of(result);
    }
    
    /**
     * 创建可学习的位置嵌入
     * 
     * @param maxLen 最大长度
     * @param dim 维度
     * @return 位置嵌入矩阵
     */
    private NdArray createLearnablePositionalEmbedding(int maxLen, int dim) {
        // 初始化为小随机值
        float[][] embedding = new float[maxLen][dim];
        java.util.Random rand = new java.util.Random(42);
        
        for (int i = 0; i < maxLen; i++) {
            for (int j = 0; j < dim; j++) {
                embedding[i][j] = (float)(rand.nextGaussian() * 0.02);
            }
        }
        
        return NdArray.of(embedding);
    }
    
    @Override
    public Variable layerForward(Variable... inputs) {
        // 简化的前向传播接口
        // 假设 input是token IDs
        LanguageInput languageInput = new LanguageInput("dummy");
        languageInput.setTokenIds(inputs[0].getValue());
        
        NdArray output = encode(languageInput);
        return new Variable(output);
    }
    
    /**
     * 简化的Transformer编码器层
     */
    private static class TransformerEncoderLayer extends Block {
        private final int hiddenDim;
        private final int numHeads;
        
        private LinearLayer qProj;
        private LinearLayer kProj;
        private LinearLayer vProj;
        private LinearLayer outProj;
        
        private LinearLayer ffn1;
        private LinearLayer ffn2;
        
        private LayerNorm norm1;
        private LayerNorm norm2;
        
        public TransformerEncoderLayer(int hiddenDim, int numHeads) {
            super("TransformerEncoderLayer", null);
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            
            int headDim = hiddenDim / numHeads;
            
            // 多头注意力投影
            this.qProj = new LinearLayer("qProj", hiddenDim, hiddenDim, false);
            this.kProj = new LinearLayer("kProj", hiddenDim, hiddenDim, false);
            this.vProj = new LinearLayer("vProj", hiddenDim, hiddenDim, false);
            this.outProj = new LinearLayer("outProj", hiddenDim, hiddenDim, false);
            
            // 前馈网络
            this.ffn1 = new LinearLayer("ffn1", hiddenDim, hiddenDim * 4, true);
            this.ffn2 = new LinearLayer("ffn2", hiddenDim * 4, hiddenDim, true);
            
            // 层归一化
            this.norm1 = new LayerNorm("norm1", hiddenDim);
            this.norm2 = new LayerNorm("norm2", hiddenDim);
        }
        
        @Override
        public void init() {
            // 初始化已在构造函数中完成
        }
        
        @Override
        public Variable layerForward(Variable... inputs) {
            Variable input = inputs[0];
            // 自注意力 + 残差连接
            Variable normed1 = norm1.layerForward(input);
            Variable attnOut = selfAttention(normed1);
            Variable residual1 = new Variable(input.getValue().add(attnOut.getValue()));
            
            // 前馈网络 + 残差连接
            Variable normed2 = norm2.layerForward(residual1);
            Variable ffn1Out = ffn1.layerForward(normed2);
            Variable reluOut = ffn1Out.relu();
            Variable ffn2Out = ffn2.layerForward(reluOut);
            Variable residual2 = new Variable(residual1.getValue().add(ffn2Out.getValue()));
            
            return residual2;
        }
        
        private Variable selfAttention(Variable input) {
            // 简化的自注意力实现
            Variable q = qProj.layerForward(input);
            Variable k = kProj.layerForward(input);
            Variable v = vProj.layerForward(input);
            
            // 计算注意力分数（简化版）
            // scores = q @ k^T / sqrt(d_k)
            NdArray qData = q.getValue();
            NdArray kData = k.getValue();
            NdArray vData = v.getValue();
            
            int seqLen = qData.getShape().getDimension(0);
            int dim = qData.getShape().getDimension(1);
            
            // 简化：直接返回value的线性变换
            Variable output = outProj.layerForward(v);
            
            return output;
        }
    }
}
