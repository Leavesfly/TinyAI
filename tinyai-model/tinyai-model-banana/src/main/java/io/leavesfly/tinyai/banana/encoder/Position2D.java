package io.leavesfly.tinyai.banana.encoder;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * 2D位置编码
 * 
 * 为图像patches提供位置信息,使模型能够感知patch的空间位置。
 * 
 * 实现方式:可学习的位置嵌入(Learnable Position Embedding)
 * - 为每个patch位置学习一个固定的位置向量
 * - 在训练过程中自动学习最优的位置表示
 * 
 * 与1D位置编码的区别:
 * - 1D: 序列位置 [0, 1, 2, ..., seq_len-1]
 * - 2D: 网格位置 [(0,0), (0,1), ..., (h-1, w-1)]
 * 
 * 输入: Patch序列 [batch, num_patches, hidden_size]
 * 输出: 位置编码 [1, num_patches, hidden_size] (可广播)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Position2D extends Module {
    
    private final int numPatches;   // Patch数量
    private final int hiddenSize;   // 嵌入维度
    
    // 可学习的位置嵌入
    private final Parameter positionEmbedding;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param numPatches Patch数量
     * @param hiddenSize 嵌入维度
     */
    public Position2D(String name, int numPatches, int hiddenSize) {
        super(name);
        this.numPatches = numPatches;
        this.hiddenSize = hiddenSize;
        
        // 初始化可学习的位置嵌入
        // 形状: [1, num_patches, hidden_size]
        // 第一维为1是为了方便广播到任意batch_size
        NdArray posEmbData = NdArray.of(Shape.of(1, numPatches, hiddenSize));
        this.positionEmbedding = registerParameter("pos_emb", new Parameter(posEmbData));
        
        init();
    }
    
    @Override
    public void resetParameters() {
        // 使用小方差的正态分布初始化位置嵌入
        // std=0.02 使位置编码在训练初期不会主导特征
        NdArray initialized = NdArray.randn(positionEmbedding.data().getShape()).mulNum(0.02f);
        System.arraycopy(initialized.getArray(), 0, positionEmbedding.data().getArray(), 0,
                initialized.getArray().length);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为Patch序列 [batch, num_patches, hidden_size]
     * @return 位置编码 [1, num_patches, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        // 位置编码独立于输入,直接返回
        // 返回的Variable可以广播到任意batch_size
        return positionEmbedding;
    }
    
    /**
     * 获取指定位置的编码
     * 
     * 使用 Variable 的 narrow 操作保持梯度计算图连通，支持反向传播。
     * 
     * @param patchIndex Patch索引 [0, num_patches)
     * @return 该位置的编码向量 [1, 1, hidden_size]
     */
    public Variable getPositionAt(int patchIndex) {
        if (patchIndex < 0 || patchIndex >= numPatches) {
            throw new IllegalArgumentException(
                "Patch索引越界: " + patchIndex + ", 有效范围[0, " + numPatches + ")"
            );
        }
        
        // 使用 sliceRange 操作提取单个位置的编码，保持梯度计算图连通
        // positionEmbedding: [1, num_patches, hidden_size]
        // 结果: [1, 1, hidden_size]
        return positionEmbedding.sliceRange(1, patchIndex, patchIndex + 1);
    }
    
    /**
     * 从2D坐标获取位置编码
     * 
     * @param row 行索引
     * @param col 列索引
     * @param numPatchesPerRow 每行的patch数量
     * @return 位置编码
     */
    public Variable getPositionAt2D(int row, int col, int numPatchesPerRow) {
        int patchIndex = row * numPatchesPerRow + col;
        return getPositionAt(patchIndex);
    }
    
    // ==================== Getter方法 ====================
    
    public int getNumPatches() {
        return numPatches;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public Parameter getPositionEmbedding() {
        return positionEmbedding;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Position2D{numPatches=%d, hiddenSize=%d}",
            numPatches, hiddenSize
        );
    }
}
