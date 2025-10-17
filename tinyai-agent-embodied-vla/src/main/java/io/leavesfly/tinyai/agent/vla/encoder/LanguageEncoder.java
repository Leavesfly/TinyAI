package io.leavesfly.tinyai.agent.vla.encoder;

import io.leavesfly.tinyai.agent.vla.model.LanguageInput;
import io.leavesfly.tinyai.agent.vla.utils.Tokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.block.Block;
import io.leavesfly.tinyai.nnet.layer.Linear;
import io.leavesfly.tinyai.nnet.layer.Embedding;
import io.leavesfly.tinyai.nnet.layer.LayerNorm;

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
        this.vocabSize = vocabSize;
        this.hiddenDim = hiddenDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        
        // 初始化Token嵌入
        this.tokenEmbedding = new Embedding(vocabSize, hiddenDim);
        
        // 初始化位置嵌入（可学习）
        this.positionalEmbedding = createLearnablePositionalEmbedding(maxSeqLen, hiddenDim);
        
        // 初始化Transformer层
        this.transformerLayers = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            transformerLayers.add(new TransformerEncoderLayer(hiddenDim, 8));
        }
        
        // 层归一化
        this.layerNorm = new LayerNorm(hiddenDim);
        
        // 初始化分词器
        this.tokenizer = new Tokenizer();
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
        NdArray tokenIdsArray = new NdArray(tokenIdsDouble).reshape(seqLen, 1);
        
        // Token嵌入
        Variable tokenEmbedVar = tokenEmbedding.forward(new Variable(tokenIdsArray, true));
        NdArray tokenEmbed = tokenEmbedVar.getData();
        
        // 添加位置嵌入
        NdArray posEmbed = positionalEmbedding.slice(new int[]{0, 0}, new int[]{seqLen, hiddenDim});
        NdArray embeddings = tokenEmbed.add(posEmbed);
        
        // 通过Transformer层
        Variable hidden = new Variable(embeddings, true);
        for (TransformerEncoderLayer layer : transformerLayers) {
            hidden = layer.forward(hidden);
        }
        
        // 最终层归一化
        Variable output = layerNorm.forward(hidden);
        
        // 保存到languageInput
        languageInput.setTokenIds(tokenIdsArray);
        languageInput.setEmbeddings(output.getData());
        
        return output.getData();
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
        double[][] embedding = new double[maxLen][dim];
        java.util.Random rand = new java.util.Random(42);
        
        for (int i = 0; i < maxLen; i++) {
            for (int j = 0; j < dim; j++) {
                embedding[i][j] = (rand.nextGaussian() * 0.02);
            }
        }
        
        return new NdArray(embedding);
    }
    
    @Override
    public Variable forward(Variable input) {
        // 简化的前向传播接口
        // 假设input是token IDs
        LanguageInput languageInput = new LanguageInput("dummy");
        languageInput.setTokenIds(input.getData());
        
        NdArray output = encode(languageInput);
        return new Variable(output, input.requiresGrad());
    }
    
    @Override
    public List<Variable> parameters() {
        List<Variable> params = new ArrayList<>();
        params.addAll(tokenEmbedding.parameters());
        params.add(new Variable(positionalEmbedding, true));
        
        for (TransformerEncoderLayer layer : transformerLayers) {
            params.addAll(layer.parameters());
        }
        
        params.addAll(layerNorm.parameters());
        return params;
    }
    
    /**
     * 简化的Transformer编码器层
     */
    private static class TransformerEncoderLayer extends Block {
        private final int hiddenDim;
        private final int numHeads;
        
        private Linear qProj;
        private Linear kProj;
        private Linear vProj;
        private Linear outProj;
        
        private Linear ffn1;
        private Linear ffn2;
        
        private LayerNorm norm1;
        private LayerNorm norm2;
        
        public TransformerEncoderLayer(int hiddenDim, int numHeads) {
            this.hiddenDim = hiddenDim;
            this.numHeads = numHeads;
            
            int headDim = hiddenDim / numHeads;
            
            // 多头注意力投影
            this.qProj = new Linear(hiddenDim, hiddenDim, false);
            this.kProj = new Linear(hiddenDim, hiddenDim, false);
            this.vProj = new Linear(hiddenDim, hiddenDim, false);
            this.outProj = new Linear(hiddenDim, hiddenDim, false);
            
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
            Variable attnOut = selfAttention(normed1);
            Variable residual1 = new Variable(input.getData().add(attnOut.getData()), input.requiresGrad());
            
            // 前馈网络 + 残差连接
            Variable normed2 = norm2.forward(residual1);
            Variable ffn1Out = ffn1.forward(normed2);
            Variable reluOut = ffn1Out.relu();
            Variable ffn2Out = ffn2.forward(reluOut);
            Variable residual2 = new Variable(residual1.getData().add(ffn2Out.getData()), input.requiresGrad());
            
            return residual2;
        }
        
        private Variable selfAttention(Variable input) {
            // 简化的自注意力实现
            Variable q = qProj.forward(input);
            Variable k = kProj.forward(input);
            Variable v = vProj.forward(input);
            
            // 计算注意力分数（简化版）
            // scores = q @ k^T / sqrt(d_k)
            NdArray qData = q.getData();
            NdArray kData = k.getData();
            NdArray vData = v.getData();
            
            int seqLen = qData.getShape()[0];
            int dim = qData.getShape()[1];
            
            // 简化：直接返回value的线性变换
            Variable output = outProj.forward(v);
            
            return output;
        }
        
        @Override
        public List<Variable> parameters() {
            List<Variable> params = new ArrayList<>();
            params.addAll(qProj.parameters());
            params.addAll(kProj.parameters());
            params.addAll(vProj.parameters());
            params.addAll(outProj.parameters());
            params.addAll(ffn1.parameters());
            params.addAll(ffn2.parameters());
            params.addAll(norm1.parameters());
            params.addAll(norm2.parameters());
            return params;
        }
    }
}
