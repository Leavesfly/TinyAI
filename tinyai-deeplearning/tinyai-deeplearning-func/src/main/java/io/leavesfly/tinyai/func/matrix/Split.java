package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 数组切分 (Split)
 */
public class Split extends Function {

    private final int splitSize;
    private final int dim;
    private final int index; // 当前是第几个分片

    public Split(int splitSize, int dim, int index) {
        this.splitSize = splitSize;
        this.dim = dim;
        this.index = index;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        Shape shape = x.getShape();
        int rank = shape.getDimNum();
        int actualDim = dim < 0 ? rank + dim : dim;
        
        // 计算切片的起始和结束
        int start = index * splitSize;
        int end = Math.min(start + splitSize, shape.getDimension(actualDim));
        
        // 构造切片索引
        // 目前 NdArray.subNdArray 仅支持 2D
        // 对于高维，我们需要更通用的 slice。
        // 鉴于 subNdArray 实现限制，这里可能需要 NdArray 增强。
        // 假设 subNdArray 暂时只处理 2D，或者我们假设输入已被 reshape。
        
        if (rank == 2) {
            if (actualDim == 0) {
                return x.subNdArray(start, end, 0, shape.getColumn());
            } else {
                return x.subNdArray(0, shape.getRow(), start, end);
            }
        } else {
            // 暂时抛出异常，等待 subNdArray 升级支持高维
            throw new UnsupportedOperationException("Split currently only supports 2D tensors.");
        }
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        Shape shape = x.getShape();
        int rank = shape.getDimNum();
        int actualDim = dim < 0 ? rank + dim : dim;
        int start = index * splitSize;
        int end = Math.min(start + splitSize, shape.getDimension(actualDim));

        NdArray grad = NdArray.zeros(shape);

        if (rank == 2) {
            float[][] gradMatrix = grad.getMatrix();
            float[][] yGradMatrix = yGrad.getMatrix();

            if (actualDim == 0) {
                for (int i = start; i < end; i++) {
                    System.arraycopy(yGradMatrix[i - start], 0, gradMatrix[i], 0, shape.getColumn());
                }
            } else {
                for (int i = 0; i < shape.getRow(); i++) {
                    System.arraycopy(yGradMatrix[i], 0, gradMatrix[i], start, end - start);
                }
            }
        } else {
            throw new UnsupportedOperationException("Split backward currently only supports 2D tensors.");
        }

        return Collections.singletonList(grad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

