package io.leavesfly.tinyai.func.util;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 广播操作工具类
 * <p>
 * 提供广播判断和梯度形状还原的公共方法，
 * 供 Add、Sub、Mul、Div、MatMul、Transpose 等函数复用。
 */
public final class BroadcastUtils {

    private BroadcastUtils() {
        // 工具类禁止实例化
    }

    /**
     * 判断源形状是否可以广播到目标形状
     * <p>
     * 广播规则（从后往前逐维比较）：
     * - 源维度等于目标维度，或者源维度为1
     * - 源的维度数不超过目标的维度数
     *
     * @param srcShape 源形状
     * @param dstShape 目标形状
     * @return 是否可以广播
     */
    public static boolean isBroadcastable(Shape srcShape, Shape dstShape) {
        if (srcShape.getDimNum() > dstShape.getDimNum()) {
            return false;
        }
        for (int i = 0; i < srcShape.getDimNum(); i++) {
            int srcDimIndex = srcShape.getDimNum() - 1 - i;
            int dstDimIndex = dstShape.getDimNum() - 1 - i;

            int srcDim = srcShape.getDimension(srcDimIndex);
            int dstDim = dstShape.getDimension(dstDimIndex);

            if (srcDim != dstDim && srcDim != 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * 将梯度 sumTo 回目标形状，支持维度数不同的情况
     * <p>
     * 当目标维度数小于梯度维度数时，先将目标形状前面补1扩展到相同维度数，
     * 执行 sumTo 后再 reshape 回原始目标形状。
     *
     * @param grad        梯度张量
     * @param targetShape 目标形状
     * @param gradShape   梯度的当前形状
     * @return 还原到目标形状的梯度
     */
    public static NdArray sumToShape(NdArray grad, Shape targetShape, Shape gradShape) {
        int targetNdim = targetShape.getDimNum();
        int gradNdim = gradShape.getDimNum();

        // 单元素目标：对所有元素求和，并 reshape 回目标形状
        if (targetShape.size() == 1) {
            float sumValue = grad.sum().getNumber().floatValue();
            return NdArray.of(new float[]{sumValue}, targetShape);
        }

        if (targetNdim < gradNdim) {
            int[] targetDims = targetShape.getShapeDims();
            int[] expandedDims = new int[gradNdim];
            int offset = gradNdim - targetNdim;
            for (int i = 0; i < offset; i++) {
                expandedDims[i] = 1;
            }
            for (int i = 0; i < targetNdim; i++) {
                expandedDims[offset + i] = targetDims[i];
            }
            Shape expandedShape = Shape.of(expandedDims);
            NdArray result = grad.sumTo(expandedShape);
            return result.reshape(targetShape);
        } else {
            return grad.sumTo(targetShape);
        }
    }

    /**
     * 将梯度 sumTo 回目标形状（自动获取梯度形状）
     *
     * @param grad        梯度张量
     * @param targetShape 目标形状
     * @return 还原到目标形状的梯度
     */
    public static NdArray sumToShape(NdArray grad, Shape targetShape) {
        return sumToShape(grad, targetShape, grad.getShape());
    }
}
