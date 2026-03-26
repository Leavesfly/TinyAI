package io.leavesfly.tinyai.minimind.model.transformer.attention;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * KV-Cache 增量推理缓存管理
 * <p>
 * 功能：
 * - 缓存历史 Key、Value 向量，避免重复计算
 * - 支持增量添加新 token 的 K、V
 * - 动态维护缓存序列长度
 * <p>
 * 应用场景：
 * - 自回归文本生成
 * - 减少重复的注意力计算开销
 *
 * @author leavesfly
 * @version 1.0
 */
public class KVCache {

    /**
     * 缓存的 Key 向量
     * Shape: [batchSize, numHeads, maxCacheLen, headDim]
     * 使用预分配策略，避免每次更新都创建新数组
     */
    private NdArray cachedK;

    /**
     * 缓存的 Value 向量
     * Shape: [batchSize, numHeads, maxCacheLen, headDim]
     * 使用预分配策略，避免每次更新都创建新数组
     */
    private NdArray cachedV;

    /**
     * 当前缓存的序列长度
     */
    private int currentSeqLen;

    /**
     * 批次大小
     */
    private final int batchSize;

    /**
     * 注意力头数
     */
    private final int numHeads;

    /**
     * 每个头的维度
     */
    private final int headDim;

    /**
     * 最大缓存长度
     */
    private final int maxCacheLen;

    /**
     * 构造 KVCache
     *
     * @param batchSize   批次大小
     * @param numHeads    注意力头数
     * @param headDim     每个头的维度
     * @param maxCacheLen 最大缓存序列长度
     */
    public KVCache(int batchSize, int numHeads, int headDim, int maxCacheLen) {
        this.batchSize = batchSize;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxCacheLen = maxCacheLen;
        this.currentSeqLen = 0;
        
        // 优化: 预分配最大长度的缓冲区，避免每次更新都创建新数组
        this.cachedK = NdArray.zeros(Shape.of(batchSize, numHeads, maxCacheLen, headDim));
        this.cachedV = NdArray.zeros(Shape.of(batchSize, numHeads, maxCacheLen, headDim));
    }

    /**
     * 更新缓存：添加新的 K、V
     *
     * @param newK 新的 Key 向量，Shape: [batchSize, numHeads, newSeqLen, headDim]
     * @param newV 新的 Value 向量，Shape: [batchSize, numHeads, newSeqLen, headDim]
     * @return 更新后的完整 K、V 数组
     */
    public NdArray[] update(NdArray newK, NdArray newV) {
        int[] newShape = newK.getShape().getShapeDims();
        int newSeqLen = newShape[2];
        
        // 检查是否会超出最大长度
        if (currentSeqLen + newSeqLen > maxCacheLen) {
            // 滑动窗口：移除旧数据
            int excessLen = (currentSeqLen + newSeqLen) - maxCacheLen;
            shiftCache(excessLen);
            currentSeqLen -= excessLen;
        }
        
        // 优化: 直接写入预分配位置，避免创建新数组
        copyToBuffer(cachedK, newK, currentSeqLen);
        copyToBuffer(cachedV, newV, currentSeqLen);
        
        currentSeqLen += newSeqLen;
        
        // 返回当前有效部分
        return new NdArray[]{
            sliceSeqDim(cachedK, 0, currentSeqLen),
            sliceSeqDim(cachedV, 0, currentSeqLen)
        };
    }
    
    /**
     * 将新数据拷贝到预分配缓冲区的指定位置
     * 
     * @param buffer 预分配的缓冲区
     * @param newData 新数据
     * @param offset 写入位置的偏移量
     */
    private void copyToBuffer(NdArray buffer, NdArray newData, int offset) {
        float[] bufferData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) buffer).buffer;
        float[] newDataArr = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) newData).buffer;
        
        int[] newShape = newData.getShape().getShapeDims();
        int newSeqLen = newShape[2];
        
        // 边界检查：验证 offset + newSeqLen 不超过 maxCacheLen
        if (offset + newSeqLen > maxCacheLen) {
            throw new IllegalArgumentException(
                String.format("Buffer overflow: offset=%d + newSeqLen=%d exceeds maxCacheLen=%d", 
                             offset, newSeqLen, maxCacheLen));
        }
        
        // 优化：使用 System.arraycopy 批量操作，减少循环嵌套
        int batchHeadStride = maxCacheLen * headDim;
        int seqStride = headDim;
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                int batchHeadOffset = (b * numHeads + h) * batchHeadStride + offset * seqStride;
                int newBatchHeadOffset = (b * numHeads + h) * newSeqLen * seqStride;
                
                // 批量拷贝整个序列的 headDim 维度
                for (int s = 0; s < newSeqLen; s++) {
                    int srcPos = newBatchHeadOffset + s * seqStride;
                    int dstPos = batchHeadOffset + s * seqStride;
                    System.arraycopy(newDataArr, srcPos, bufferData, dstPos, headDim);
                }
            }
        }
    }
    
    /**
     * 滑动窗口：移除旧数据
     * 
     * @param shiftLen 需要移除的序列长度
     */
    private void shiftCache(int shiftLen) {
        // 将数据向前移动 shiftLen 个位置
        shiftBuffer(cachedK, shiftLen);
        shiftBuffer(cachedV, shiftLen);
    }
    
    /**
     * 移动缓冲区数据
     * 
     * @param buffer 缓冲区
     * @param shiftLen 移动长度
     */
    private void shiftBuffer(NdArray buffer, int shiftLen) {
        float[] bufferData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) buffer).buffer;
        
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = shiftLen; s < maxCacheLen; s++) {
                    for (int d = 0; d < headDim; d++) {
                        int srcIdx = ((b * numHeads + h) * maxCacheLen + s) * headDim + d;
                        int dstIdx = ((b * numHeads + h) * maxCacheLen + (s - shiftLen)) * headDim + d;
                        bufferData[dstIdx] = bufferData[srcIdx];
                    }
                }
            }
        }
    }

    /**
     * 在序列维度上切片
     *
     * @param data 原始数据
     * @param start 起始位置
     * @param end 结束位置
     * @return 切片后的数组
     */
    private NdArray sliceSeqDim(NdArray data, int start, int end) {
        int[] shape = data.getShape().getShapeDims();
        int batch = shape[0];
        int heads = shape[1];
        int seqLen = shape[2];
        int dim = shape[3];

        int newSeqLen = end - start;
        float[] result = new float[batch * heads * newSeqLen * dim];
        float[] srcData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) data).buffer;

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int s = start; s < end; s++) {
                    for (int d = 0; d < dim; d++) {
                        int srcIdx = ((b * heads + h) * seqLen + s) * dim + d;
                        int dstIdx = ((b * heads + h) * newSeqLen + (s - start)) * dim + d;
                        result[dstIdx] = srcData[srcIdx];
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batch, heads, newSeqLen, dim));
    }

    /**
     * 清空缓存
     */
    public void clear() {
        currentSeqLen = 0;
        // 清空预分配缓冲区
        float[] kBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) cachedK).buffer;
        float[] vBuffer = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) cachedV).buffer;
        java.util.Arrays.fill(kBuffer, 0.0f);
        java.util.Arrays.fill(vBuffer, 0.0f);
    }

    /**
     * 获取当前缓存的序列长度
     */
    public int getCurrentSeqLen() {
        return currentSeqLen;
    }

    /**
     * 获取缓存的 Key
     */
    public NdArray getCachedK() {
        return sliceSeqDim(cachedK, 0, currentSeqLen);
    }

    /**
     * 获取缓存的 Value
     */
    public NdArray getCachedV() {
        return sliceSeqDim(cachedV, 0, currentSeqLen);
    }

    /**
     * 判断缓存是否为空
     */
    public boolean isEmpty() {
        return currentSeqLen == 0;
    }
}