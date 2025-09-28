package io.leavesfly.tinyai.nnet.layer.embedd;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.Layer;
import io.leavesfly.tinyai.nnet.Parameter;

import java.util.Arrays;
import java.util.List;

/**
 * 词嵌入层实现
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * Embedding层用于将离散的词汇索引转换为连续的向量表示。
 * 它维护一个词汇表大小×嵌入维度的权重矩阵，通过查找表的方式获取对应词向量。
 * <p>
 * 前向传播过程：
 * 1. 输入为词汇索引序列
 * 2. 根据索引从权重矩阵中查找对应的词向量
 * <p>
 * 反向传播过程：
 * 1. 梯度通过索引位置累加到权重矩阵对应位置
 */
public class Embedding extends Layer {
    /**
     * 嵌入权重矩阵
     * 形状: (vocabSize, embedSize)
     */
    private Parameter wIn;

    /**
     * 词汇表大小
     */
    private int vocabSize;

    /**
     * 嵌入维度
     */
    private int embedSize;

    /**
     * 构造函数，创建Embedding层实例
     *
     * @param _name     层名称
     * @param vocabSize 词汇表大小
     * @param embedSize 嵌入维度
     */
    public Embedding(String _name, int vocabSize, int embedSize) {
        super(_name, Shape.of(vocabSize), Shape.of(vocabSize, embedSize));
        this.vocabSize = vocabSize;
        this.embedSize = embedSize;
        NdArray initWeight = NdArray.likeRandomN(Shape.of(vocabSize, embedSize)).mulNum(0.01f);
        wIn = new Parameter(initWeight);
        wIn.setName("wIn");
        addParam(wIn.getName(), wIn);
    }

    @Override
    public void init() {
        // Embedding层不需要额外的初始化
    }

    /**
     * 层的前向传播计算
     * <p>
     * 根据输入的词汇索引从权重矩阵中查找对应的词向量
     *
     * @param inputs 输入变量数组，包含词汇索引
     * @return 前向传播结果变量
     */
    @Override
    public Variable layerForward(Variable... inputs) {
        Variable input = inputs[0];
        NdArray inputValue = input.getValue();

        // 处理不同形状的输入
        if (inputValue.getShape().getDimNum() == 1) {
            // 一维输入 (序列长度,)
            int[] slices = NdArrayUtil.toInt(inputValue.getMatrix()[0]);
            return wIn.getItem(slices, null);
        } else if (inputValue.getShape().getDimNum() == 2) {
            // 二维输入 (batch_size, sequence_length)
            int batchSize = inputValue.getShape().getRow();
            int seqLength = inputValue.getShape().getColumn();
            
            // 创建结果数组: (batch_size, sequence_length, embedding_dim)
            NdArray result = NdArray.zeros(Shape.of(batchSize, seqLength, embedSize));
            
            // 为每个样本处理嵌入
            for (int i = 0; i < batchSize; i++) {
                int[] slices = NdArrayUtil.toInt(inputValue.getMatrix()[i]);
                Variable embeddedSample = wIn.getItem(slices, null);
                // 将嵌入结果复制到结果数组中
                for (int j = 0; j < seqLength; j++) {
                    for (int k = 0; k < embedSize; k++) {
                        result.set(embeddedSample.getValue().get(j, k), i, j, k);
                    }
                }
            }
            
            // 如果序列长度为1，重新形状为 (batch_size, embedding_dim)
            if (seqLength == 1) {
                result = result.reshape(Shape.of(batchSize, embedSize));
            }
            
            return new Variable(result);
        } else {
            throw new IllegalArgumentException("Embedding层不支持该输入形状: " + inputValue.getShape());
        }
    }

    /**
     * NdArray前向传播计算（未使用）
     *
     * @param inputs 输入的NdArray数组
     * @return 前向传播结果NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        // Embedding层主要通过layerForward方法处理Variable输入
        return null;
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于Embedding层，梯度通过索引位置累加到权重矩阵对应位置
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // Embedding层的输入是离散索引，不需要计算输入的梯度
        // 但需要累积输出梯度到权重矩阵的对应位置
        return Arrays.asList(NdArray.zeros(inputShape)); // 输入梯度为0
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * Embedding层需要一个输入参数：词汇索引。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 获取词汇表大小
     *
     * @return 词汇表大小
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * 获取嵌入维度
     *
     * @return 嵌入维度
     */
    public int getEmbedSize() {
        return embedSize;
    }

    /**
     * 获取嵌入权重矩阵
     *
     * @return 嵌入权重矩阵参数
     */
    public Parameter getWeight() {
        return wIn;
    }
}