package io.leavesfly.tinyai.nnet.block.seq2seq;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.layer.embedd.Embedding;
import io.leavesfly.tinyai.nnet.layer.norm.Dropout;
import io.leavesfly.tinyai.nnet.layer.rnn.LstmLayer;

/**
 * 基于LSTM的序列到序列编码器实现
 * <p>
 * 这个编码器采用了经典的RNN-based架构，由以下组件按顺序组成：
 * 1. Embedding层：将离散的词索引转换为稠密的词向量表示
 * 2. LSTM层：对序列进行建模，捕捉长距离依赖关系
 * 3. Dropout层：防止过拟合，提高模型泛化能力
 * <p>
 * 编码器的输出可以是：
 * - 最后一个时间步的隐藏状态（用于初始化解码器）
 * - 所有时间步的隐藏状态（用于注意力机制）
 * <p>
 * 使用场景：
 * - 机器翻译的源语言编码
 * - 文本摘要的原文编码
 * - 对话系统的上下文编码
 * - 问答系统的问题编码
 * <p>
 * 使用示例：
 * ```java
 * // 创建编码器
 * Seq2SeqEncoder encoder = new Seq2SeqEncoder(
 * "encoder",
 * Shape.of(-1, 50),          // 输入形状: [batch_size, seq_len]
 * Shape.of(-1, 256),         // 输出形状: [batch_size, hidden_dim]
 * 10000,                      // 词汇表大小
 * 128,                       // 嵌入维度
 * 256,                       // LSTM隐藏层大小
 * 0.1                        // Dropout比率
 * );
 * <p>
 * // 输入序列进行编码
 * Variable encoderOutput = encoder.layerForward(inputSequence);
 * ```
 *
 * @author TinyDL
 * @version 0.01
 * @since 2025-01-01
 */
public class Seq2SeqEncoder extends Encoder {

    /**
     * 词嵌入层
     */
    private Embedding embedding;

    /**
     * LSTM循环结层
     */
    private LstmLayer lstmLayer;

    /**
     * Dropout正则化层
     */
    private Dropout dropout;

    // 模型参数
    private final int vocabSize;        // 词汇表大小
    private final int embeddingDim;     // 嵌入维度
    private final int hiddenSize;       // LSTM隐藏层大小
    private final double dropoutRate;   // Dropout比率

    /**
     * 标记层是否已初始化
     */
    private boolean layersInitialized = false;

    /**
     * 构造序列到序列编码器
     *
     * @param _name         编码器名称
     * @param _xInputShape  输入形状，期望为 [batch_size, seq_length]
     * @param _yOutputShape 输出形状，期望为 [batch_size, hidden_size] 或 [batch_size, seq_length, hidden_size]
     * @param vocabSize     词汇表大小，必须大于0
     * @param embeddingDim  词嵌入维度，必须大于0
     * @param hiddenSize    LSTM隐藏层大小，必须大于0
     * @param dropoutRate   Dropout比率，应在[0, 1)范围内
     * @throws IllegalArgumentException 当输入参数无效时抛出
     */
    public Seq2SeqEncoder(String _name, Shape _xInputShape, Shape _yOutputShape,
                          int vocabSize, int embeddingDim, int hiddenSize, double dropoutRate) {
        super(_name, _xInputShape, _yOutputShape);

        // 参数验证
        validateParameters(vocabSize, embeddingDim, hiddenSize, dropoutRate);

        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenSize = hiddenSize;
        this.dropoutRate = dropoutRate;
    }

    /**
     * 替代构造函数，保持与原有代码的兼容性
     *
     * @deprecated 请使用带参数的构造函数以获取更好的初始化支持
     */
    @Deprecated
    public Seq2SeqEncoder(String _name, Shape _xInputShape, Shape _yOutputShape) {
        super(_name, _xInputShape, _yOutputShape);

        // 使用默认参数
        this.vocabSize = 10000;
        this.embeddingDim = 128;
        this.hiddenSize = 256;
        this.dropoutRate = 0.1;

        System.out.println("警告: 使用了已弃用的构造函数，建议使用带参数的构造函数");
    }

    /**
     * 验证构造参数
     */
    private void validateParameters(int vocabSize, int embeddingDim, int hiddenSize, double dropoutRate) {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0, 当前值: " + vocabSize);
        }
        if (embeddingDim <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0, 当前值: " + embeddingDim);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("LSTM隐藏层大小必须大于0, 当前值: " + hiddenSize);
        }
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw new IllegalArgumentException(
                    "Dropout比率必须在[0.0, 1.0)范围内, 当前值: " + dropoutRate);
        }
    }

    /**
     * 初始化编码器的所有层
     * <p>
     * 该方法会创建并配置所有的子层，包括嵌入层、LSTM层和Dropout层。
     */
    @Override
    public void init() {
        if (!layersInitialized) {
            try {
                // 创建词嵌入层
                this.embedding = new Embedding(
                        name + "_embedding",
                        vocabSize,
                        embeddingDim
                );

                // 创建LSTM层
                this.lstmLayer = new LstmLayer(
                        name + "_lstm",
                        Shape.of(-1, -1, embeddingDim),  // 输入形状
                        Shape.of(-1, -1, hiddenSize)     // 输出形状
                );

                // 创建Dropout层
                this.dropout = new Dropout(
                        name + "_dropout",
                        (float) dropoutRate
                );

                // 将子层添加到当前块中
                addLayer(embedding);
                addLayer(lstmLayer);
                addLayer(dropout);

                layersInitialized = true;

                System.out.println(String.format(
                        "Seq2SeqEncoder '%s' 初始化成功 - 词汇: %d, 嵌入: %d, 隐藏: %d, Dropout: %.2f",
                        name, vocabSize, embeddingDim, hiddenSize, dropoutRate
                ));

            } catch (Exception e) {
                throw new RuntimeException(String.format(
                        "Seq2SeqEncoder '%s' 初始化失败: %s", name, e.getMessage()), e);
            }
        }
    }

    /**
     * 执行编码器的前向传播
     * <p>
     * 该方法实现了完整的编码流程：词嵌入 -> LSTM处理 -> Dropout正则化。
     *
     * @param inputs 输入参数，期望包含一个参数：
     *               inputs[0] - 输入序列，形状为 [batch_size, seq_length]
     * @return 编码后的表示，形状为 [batch_size, seq_length, hidden_size] 或 [batch_size, hidden_size]
     * @throws IllegalArgumentException 当输入参数不正确时抛出
     * @throws IllegalStateException    当编码器尚未初始化时抛出
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        // 输入验证
        validateForwardInputs(inputs);

        // 确保层已初始化
        if (!layersInitialized) {
            init();
        }

        try {
            Variable input = inputs[0];

            // 第一步：词嵌入
            Variable y = embedding.layerForward(input);

            // 第二步：LSTM处理
            y = lstmLayer.layerForward(y);

            // 第三步：Dropout正则化
            y = dropout.layerForward(y);

            return y;

        } catch (Exception e) {
            throw new RuntimeException(String.format(
                    "Seq2SeqEncoder '%s' 前向传播失败: %s", name, e.getMessage()), e);
        }
    }

    /**
     * 验证前向传播的输入参数
     */
    private void validateForwardInputs(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("编码器需要至少1个输入参数");
        }
        if (inputs[0] == null) {
            throw new IllegalArgumentException("输入序列不能为null");
        }
    }

    /**
     * 获取LSTM层的最终隐藏状态
     *
     * @return LSTM的最终隐藏状态，如果LSTM层尚未初始化则返回null
     */
    @Override
    public NdArray getFinalHiddenState() {
        // 注意：当前的RnnLayer基类没有getHiddenState方法
        // 这里返回null，子类可以根据具体的RNN实现重写此方法
        return null;
    }

    /**
     * 重置编码器的内部状态
     */
    @Override
    public void resetState() {
        super.resetState();
        if (lstmLayer != null) {
            lstmLayer.resetState();
        }
    }

    // Getter方法

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取嵌入维度
     */
    public int getEmbeddingDim() {
        return embeddingDim;
    }

    /**
     * 获取LSTM隐藏层大小
     */
    public int getHiddenSize() {
        return hiddenSize;
    }

    /**
     * 获取Dropout比率
     */
    public double getDropoutRate() {
        return dropoutRate;
    }

    /**
     * 获取嵌入层实例
     */
    public Embedding getEmbedding() {
        return embedding;
    }

    /**
     * 获取LSTM层实例
     */
    public LstmLayer getLstmLayer() {
        return lstmLayer;
    }

    /**
     * 获取Dropout层实例
     */
    public Dropout getDropout() {
        return dropout;
    }

    /**
     * 检查层是否已初始化
     */
    public boolean isLayersInitialized() {
        return layersInitialized;
    }

    /**
     * 获取编码器的详细信息
     */
    @Override
    public String toString() {
        return String.format(
                "Seq2SeqEncoder(name='%s', vocab=%d, embed=%d, hidden=%d, dropout=%.2f, initialized=%s)",
                name, vocabSize, embeddingDim, hiddenSize, dropoutRate, layersInitialized
        );
    }
}
