package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.deepseek.base.DeepSeekTokenEmbeddingBase;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;

/**
 * DeepSeek-V3 Token嵌入层
 * 
 * 提供token嵌入和位置嵌入功能，将离散的token ID转换为连续的向量表示。
 * 
 * 组件：
 * 1. Token Embedding - 将token ID映射到嵌入向量
 * 2. Position Embedding - 为每个位置添加位置信息
 * 3. Dropout - 防止过拟合
 * 
 * 输出 = Dropout(TokenEmbed + PositionEmbed)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3TokenEmbedding extends DeepSeekTokenEmbeddingBase {
    
    private final DeepSeekV3Config config;
    
    // 嵌入参数
    private Parameter tokenEmbeddings;    // [vocabSize, nEmbd]
    private Parameter positionEmbeddings; // [nPositions, nEmbd]
    
    // Dropout层
    private Dropout dropout;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3TokenEmbedding(String name, DeepSeekV3Config config) {
        super(name,
                config.getVocabSize(),
                config.getNEmbd(),
                config.getNPositions(),
                (float) config.getEmbdPdrop(),
                config.getInitializerRange());
        // 注意：this.config 必须在 super() 之后赋值，
        // 而 initializeEmbeddings() 已由父类构造函数调用（通过多态），
        // 因此此处不再重复调用。
        this.config = config;
    }
    
    @Override
    protected void initializeEmbeddings() {
        // 父类构造函数在 this.config 赋值之前就通过多态调用此方法，
        // 因此这里必须使用父类已赋值的字段，而不能使用 config。
        // 1. 初始化Token嵌入矩阵 [vocabSize, nEmbd]
        NdArray tokenEmbedData = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim))
                .mulNum((float) initializerRange);
        tokenEmbeddings = new Parameter(tokenEmbedData);
        registerParameter("token_embedding", tokenEmbeddings);
        
        // 2. 初始化位置嵌入矩阵 [nPositions, nEmbd]
        NdArray positionEmbedData = NdArray.likeRandomN(Shape.of(maxPositions, embeddingDim))
                .mulNum((float) initializerRange);
        positionEmbeddings = new Parameter(positionEmbedData);
        registerParameter("position_embedding", positionEmbeddings);
        
        // 3. 初始化Dropout层
        dropout = new Dropout("embedding_dropout", dropoutProb);
        registerModule("dropout", dropout);
    }
    
    /**
     * 初始化权重矩阵（使用正态分布）
     */
    private void initializeWeights(float[][] weights, double std) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                // 使用简单的随机初始化（实际应用中应使用正态分布）
                weights[i][j] = (float) ((Math.random() - 0.5) * 2 * std);
            }
        }
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为token ID序列 [batch_size, seq_len]
     * @return 嵌入向量 [batch_size, seq_len, nEmbd]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];
        NdArray tokenIdsArray = tokenIds.getValue();
        
        // 验证输入维度
        if (tokenIdsArray.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    tokenIdsArray.getShape())
            );
        }
        
        int batchSize = tokenIdsArray.getShape().getDimension(0);
        int seqLen = tokenIdsArray.getShape().getDimension(1);
        
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions())
            );
        }
        
        // ✅ 使用Variable层面的算子，保持在Variable World
        // 1. 获取Token嵌入 - 使用indexSelect算子
        Variable tokenEmbedParam = new Variable(tokenEmbeddings.data());
        Variable tokenEmbeds = getTokenEmbeddingsV2(tokenIds, tokenEmbedParam, batchSize, seqLen);
        
        // 2. 获取位置嵌入 - 使用indexSelect算子
        Variable posEmbedParam = new Variable(positionEmbeddings.data());
        Variable positionEmbeds = getPositionEmbeddingsV2(posEmbedParam, batchSize, seqLen);
        
        // 3. 相加并应用dropout
        Variable combined = tokenEmbeds.add(positionEmbeds);
        return dropout.forward(combined);
    }
    
    /**
     * 获取Token嵌入 (使用Variable算子)
     * 
     * @param tokenIds token ID变量 [batch_size, seq_len]
     * @param tokenEmbedParam token嵌入参数 [vocabSize, nEmbd]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return token嵌入 [batch_size, seq_len, nEmbd]
     */
    @Override
    protected Variable getTokenEmbeddingsV2(Variable tokenIds, Variable tokenEmbedParam, 
                                          int batchSize, int seqLen) {
        // ✅ 使用indexSelect算子在Variable层面操作
        // tokenEmbedParam: [vocabSize, nEmbd]
        // tokenIds: [batch_size, seq_len]
        // 需要将tokenIds展平为1D，然后indexSelect，最后reshape回3D
        
        // 1. 展平tokenIds: [batch_size, seq_len] -> [batch_size * seq_len]
        Variable flatTokenIds = tokenIds.reshape(Shape.of(batchSize * seqLen));
        
        // 2. 使用indexSelect选择嵌入: [batch_size * seq_len, nEmbd]
        Variable flatEmbeds = tokenEmbedParam.indexSelect(0, flatTokenIds);
        
        // 3. Reshape回3D: [batch_size, seq_len, nEmbd]
        return flatEmbeds.reshape(Shape.of(batchSize, seqLen, config.getNEmbd()));
    }
    
    /**
     * 获取位置嵌入 (使用Variable算子)
     * 
     * @param posEmbedParam 位置嵌入参数 [nPositions, nEmbd]
     * @param batchSize 批大小
     * @param seqLen 序列长度
     * @return 位置嵌入 [1, seqLen, nEmbd] - 依赖广播机制自动扩展
     */
    @Override
    protected Variable getPositionEmbeddingsV2(Variable posEmbedParam, int batchSize, int seqLen) {
        // ✅ 使用indexSelect算子，不使用repeat以节省内存
        // 返回 [1, seqLen, nEmbd] 形状，依赖add的广播机制自动扩展
        float[] posIndices = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            posIndices[i] = i;
        }
        Variable posIds = new Variable(NdArray.of(posIndices));
        posIds.setRequireGrad(false);
        Variable posEmbeds = posEmbedParam.indexSelect(0, posIds);
        return posEmbeds.reshape(Shape.of(1, seqLen, config.getNEmbd()));
    }
    
    /**
     * 获取配置对象
     */
    public DeepSeekV3Config getConfig() {
        return config;
    }
    
    /**
     * 获取嵌入维度
     */
    public int getEmbeddingDim() {
        return config.getNEmbd();
    }
    
    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return config.getVocabSize();
    }
    
    /**
     * 获取最大位置数
     */
    public int getMaxPositions() {
        return config.getNPositions();
    }
}
