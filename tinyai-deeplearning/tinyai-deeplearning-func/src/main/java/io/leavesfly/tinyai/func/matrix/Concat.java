package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * 拼接算子 (Concatenate)
 * <p>
 * forward: 将多个 Variable 沿指定维度拼接
 * backward: 将梯度沿指定维度切分回传
 */
public class Concat extends Function {

    private final int dim;
    private int[] splitSizes;
    private Shape[] inputShapes;

    public Concat(int dim) {
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("Concat inputs cannot be empty");
        }
        
        // 记录输入形状以便反向传播
        inputShapes = new Shape[inputs.length];
        splitSizes = new int[inputs.length];
        
        int totalSizeInDim = 0;
        Shape baseShape = inputs[0].getShape();
        int targetDim = dim < 0 ? baseShape.getDimNum() + dim : dim;
        
        for (int i = 0; i < inputs.length; i++) {
            inputShapes[i] = inputs[i].getShape();
            splitSizes[i] = inputShapes[i].getDimension(targetDim);
            totalSizeInDim += splitSizes[i];
            
            // 校验其他维度是否一致
            if (i > 0) {
                // 简单校验维度数
                if (inputShapes[i].getDimNum() != baseShape.getDimNum()) {
                    throw new IllegalArgumentException("Input shapes must have same rank");
                }
            }
        }
        
        // 计算目标形状
        int[] newDims = baseShape.getShapeDims().clone();
        newDims[targetDim] = totalSizeInDim;
        Shape targetShape = Shape.of(newDims);
        
        // 创建结果数组
        NdArray result = NdArray.zeros(targetShape);
        
        int currentOffset = 0;
        
        // 基于行优先内存布局的通用拼接实现，支持任意维度
        float[] resultData = result.getArray();
        int rank = baseShape.getDimNum();
        
        if (rank == 2) {
            // 2D 快速路径：使用 setBlock 高效赋值
            for (int i = 0; i < inputs.length; i++) {
                NdArray input = inputs[i];
                int rows = input.getShape().getRow();
                int cols = input.getShape().getColumn();
                
                if (targetDim == 0) {
                    result.setBlock(currentOffset, currentOffset + rows, 0, cols, input.getArray());
                    currentOffset += rows;
                } else {
                    result.setBlock(0, rows, currentOffset, currentOffset + cols, input.getArray());
                    currentOffset += cols;
                }
            }
        } else {
            // 通用路径：支持 3D 及更高维度
            // 行优先布局中，将维度分为 outer（targetDim 之前）、concat（targetDim）、inner（targetDim 之后）
            int outerSize = 1;
            for (int d = 0; d < targetDim; d++) {
                outerSize *= newDims[d];
            }
            int innerSize = 1;
            for (int d = targetDim + 1; d < rank; d++) {
                innerSize *= newDims[d];
            }
            
            for (int outer = 0; outer < outerSize; outer++) {
                int resultDimOffset = 0;
                for (int i = 0; i < inputs.length; i++) {
                    float[] inputData = inputs[i].getArray();
                    int inputDimSize = splitSizes[i];
                    int inputInnerStride = inputDimSize * innerSize;
                    
                    int srcOffset = outer * inputInnerStride;
                    int dstOffset = outer * totalSizeInDim * innerSize + resultDimOffset * innerSize;
                    
                    System.arraycopy(inputData, srcOffset, resultData, dstOffset, inputInnerStride);
                    resultDimOffset += inputDimSize;
                }
            }
        }
        
        return result;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        List<NdArray> grads = new ArrayList<>();
        int targetDim = dim < 0 ? yGrad.getShape().getDimNum() + dim : dim;
        int rank = yGrad.getShape().getDimNum();
        
        if (rank == 2) {
            // 2D 快速路径
            int currentOffset = 0;
            for (int i = 0; i < splitSizes.length; i++) {
                int size = splitSizes[i];
                int rows = inputShapes[i].getRow();
                int cols = inputShapes[i].getColumn();
                
                int startRow, endRow, startCol, endCol;
                if (targetDim == 0) {
                    startRow = currentOffset;
                    endRow = currentOffset + size;
                    startCol = 0;
                    endCol = cols;
                    currentOffset += size;
                } else {
                    startRow = 0;
                    endRow = rows;
                    startCol = currentOffset;
                    endCol = currentOffset + size;
                    currentOffset += size;
                }
                
                NdArray subGrad = yGrad.subNdArray(startRow, endRow, startCol, endCol);
                grads.add(subGrad.mulNum(1.0f));
            }
        } else {
            // 通用路径：支持 3D 及更高维度，与 forward 的通用路径对称
            int[] gradDims = yGrad.getShape().getShapeDims();
            int totalSizeInDim = gradDims[targetDim];
            
            int outerSize = 1;
            for (int d = 0; d < targetDim; d++) {
                outerSize *= gradDims[d];
            }
            int innerSize = 1;
            for (int d = targetDim + 1; d < rank; d++) {
                innerSize *= gradDims[d];
            }
            
            float[] gradData = yGrad.getArray();
            
            for (int i = 0; i < splitSizes.length; i++) {
                int inputDimSize = splitSizes[i];
                int inputTotalSize = outerSize * inputDimSize * innerSize;
                float[] splitData = new float[inputTotalSize];
                
                int dimOffset = 0;
                for (int j = 0; j < i; j++) {
                    dimOffset += splitSizes[j];
                }
                
                for (int outer = 0; outer < outerSize; outer++) {
                    int srcOffset = outer * totalSizeInDim * innerSize + dimOffset * innerSize;
                    int dstOffset = outer * inputDimSize * innerSize;
                    System.arraycopy(gradData, srcOffset, splitData, dstOffset, inputDimSize * innerSize);
                }
                
                grads.add(NdArray.of(splitData, inputShapes[i]));
            }
        }
        
        return grads;
    }

    @Override
    public int requireInputNum() {
        return -1; // Variable inputs
    }
}

