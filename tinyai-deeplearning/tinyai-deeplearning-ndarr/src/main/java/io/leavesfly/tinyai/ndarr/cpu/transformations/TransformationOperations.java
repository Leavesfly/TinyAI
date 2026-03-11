package io.leavesfly.tinyai.ndarr.cpu.transformations;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;
import io.leavesfly.tinyai.ndarr.cpu.utils.ArrayValidator;
import io.leavesfly.tinyai.ndarr.cpu.utils.IndexConverter;

/**
 * 变形操作类，提供转置、重塑、广播等形状变换功能。
 */
public class TransformationOperations {

    /**
     * 矩阵转置，二维时行列互换，多维时交换最后两个维度。
     *
     * @param array 数组
     * @return 转置后的数组
     * @throws IllegalArgumentException 当数组维度小于 2 时抛出
     */
    public static NdArrayCpu transpose(NdArrayCpu array) {
        int dimNum = array.shape.getDimNum();
        
        if (dimNum < 2) {
            throw new IllegalArgumentException("转置操作至少需要二维数组");
        }
        
        if (dimNum == 2) {
            // 二维矩阵：行列互换（原有优化逻辑）
            NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(array.shape.getColumn(), array.shape.getRow()));
            for (int i = 0; i < array.shape.getRow(); i++) {
                for (int j = 0; j < array.shape.getColumn(); j++) {
                    result.buffer[j * array.shape.getRow() + i] = array.buffer[i * array.shape.getColumn() + j];
                }
            }
            return result;
        }
        
        // 多维数组：交换最后两个维度
        int[] order = new int[dimNum];
        for (int i = 0; i < dimNum - 2; i++) {
            order[i] = i;  // 前面的维度保持不变
        }
        order[dimNum - 2] = dimNum - 1;  // 倒数第二维与最后一维交换
        order[dimNum - 1] = dimNum - 2;
        
        return transpose(array, order);
    }

    /**
     * 多维数组转置操作，按指定维度顺序重新排列
     *
     * @param array 数组
     * @param order 新的维度顺序
     * @return 转置后的数组
     * @throws IllegalArgumentException 当维度顺序无效时抛出
     */
    public static NdArrayCpu transpose(NdArrayCpu array, int... order) {
        ArrayValidator.validateTransposeOrder(order, array.shape.getDimNum());

        int[] newDimensions = new int[array.shape.dimension.length];
        for (int i = 0; i < order.length; i++) {
            newDimensions[i] = array.shape.dimension[order[i]];
        }
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(newDimensions));

        int[] indices = new int[array.shape.dimension.length];
        int totalElements = array.shape.size();

        for (int i = 0; i < totalElements; i++) {
            // 将一维索引转换为多维索引
            IndexConverter.convertToMultiIndex(i, indices, array.shape);

            // 计算转置后的索引
            int[] transposedIndices = new int[order.length];
            for (int j = 0; j < order.length; j++) {
                transposedIndices[j] = indices[order[j]];
            }

            // 复制数据
            result.set(array.buffer[i], transposedIndices);
        }
        return result;
    }

    /**
     * 数组变形操作，改变数组形状但保持元素总数不变
     *
     * @param array    数组
     * @param newShape 新的数组形状
     * @return 变形后的数组
     * @throws IllegalArgumentException 当新形状大小与原形状不匹配时抛出
     */
    public static NdArrayCpu reshape(NdArrayCpu array, Shape newShape) {
        if (array.shape.size() != newShape.size()) {
            throw new IllegalArgumentException(String.format("形状大小不匹配：%d vs %d", array.shape.size(), newShape.size()));
        }

        // 使用共享数据的视图，避免数据复制
        return new NdArrayCpu(array.buffer, newShape);
    }

    /**
     * 数组展平操作，将多维数组转换为一维行向量
     *
     * @param array 数组
     * @return 展平后的一维行向量
     */
    public static NdArrayCpu flatten(NdArrayCpu array) {
        return reshape(array, Shape.of(1, array.shape.size()));
    }

    /**
     * 将数组广播到目标形状，小数组会重复填充以匹配大数组。
     *
     * @param array       数组
     * @param targetShape 目标形状
     * @return 广播结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    public static NdArrayCpu broadcastTo(NdArrayCpu array, Shape targetShape) {
        // 检查形状兼容性
        if (array.shape.getDimNum() > targetShape.getDimNum()) {
            throw new IllegalArgumentException(String.format("源形状维度(%d)不能大于目标形状维度(%d)", array.shape.getDimNum(), targetShape.getDimNum()));
        }

        // 检查广播兼容性
        // 从后往前检查维度是否兼容
        for (int i = 0; i < array.shape.getDimNum(); i++) {
            int srcDimIndex = array.shape.getDimNum() - 1 - i;
            int dstDimIndex = targetShape.getDimNum() - 1 - i;

            int srcDim = array.shape.getDimension(srcDimIndex);
            int dstDim = targetShape.getDimension(dstDimIndex);

            // 广播规则：维度相等，或者源维度为1，或者源维度不存在（即维度数较少）
            if (srcDim != dstDim && srcDim != 1) {
                throw new IllegalArgumentException(String.format("广播不兼容：源维度[%d]=%d，目标维度[%d]=%d", srcDimIndex, srcDim, dstDimIndex, dstDim));
            }
        }

        NdArrayCpu result = new NdArrayCpu((ShapeCpu) targetShape);

        // 执行广播
        for (int i = 0; i < targetShape.size(); i++) {
            // 计算目标索引对应源索引
            int[] dstIndices = new int[targetShape.getDimNum()];
            int temp = i;
            for (int dim = targetShape.getDimNum() - 1; dim >= 0; dim--) {
                dstIndices[dim] = temp % targetShape.getDimension(dim);
                temp /= targetShape.getDimension(dim);
            }

            // 计算源索引
            int[] srcIndices = new int[array.shape.getDimNum()];
            for (int dim = 0; dim < array.shape.getDimNum(); dim++) {
                int srcDimIndex = array.shape.getDimNum() - 1 - dim;
                int dstDimIndex = targetShape.getDimNum() - 1 - dim;

                if (array.shape.getDimension(srcDimIndex) == 1) {
                    // 如果源维度为1，则索引始终为0
                    srcIndices[srcDimIndex] = 0;
                } else {
                    // 否则使用目标索引
                    srcIndices[srcDimIndex] = dstIndices[dstDimIndex];
                }
            }

            // 获取源值并设置到目标位置
            result.buffer[i] = array.buffer[array.shape.getIndex(srcIndices)];
        }

        return result;
    }

    /**
     * 按目标形状压缩累加，超出部分累加到对应位置。
     *
     * @param array       数组
     * @param targetShape 目标形状
     * @return 压缩累加结果数组
     * @throws IllegalArgumentException 当形状不合法时抛出
     */
    public static NdArrayCpu sumTo(NdArrayCpu array, Shape targetShape) {
        // 检查形状兼容性
        if (array.shape.getDimNum() < targetShape.getDimNum()) {
            throw new IllegalArgumentException(String.format("源形状维度(%d)不能小于目标形状维度(%d)", array.shape.getDimNum(), targetShape.getDimNum()));
        }

        // 检查sumTo兼容性
        // 从后往前检查维度是否兼容
        for (int i = 0; i < targetShape.getDimNum(); i++) {
            int srcDimIndex = array.shape.getDimNum() - 1 - i;
            int dstDimIndex = targetShape.getDimNum() - 1 - i;

            int srcDim = array.shape.getDimension(srcDimIndex);
            int dstDim = targetShape.getDimension(dstDimIndex);

            // sumTo规则：维度相等，或者目标维度为1，或者目标维度不存在（即维度数较少）
            if (srcDim != dstDim && dstDim != 1) {
                throw new IllegalArgumentException(String.format("sumTo不兼容：源维度[%d]=%d，目标维度[%d]=%d", srcDimIndex, srcDim, dstDimIndex, dstDim));
            }
        }

        NdArrayCpu result = new NdArrayCpu((ShapeCpu) targetShape);

        // 执行sumTo
        for (int i = 0; i < array.shape.size(); i++) {
            // 计算源索引
            int[] srcIndices = new int[array.shape.getDimNum()];
            int temp = i;
            for (int dim = array.shape.getDimNum() - 1; dim >= 0; dim--) {
                srcIndices[dim] = temp % array.shape.getDimension(dim);
                temp /= array.shape.getDimension(dim);
            }

            // 计算目标索引
            int[] dstIndices = new int[targetShape.getDimNum()];
            boolean valid = true;

            for (int dim = 0; dim < targetShape.getDimNum(); dim++) {
                int dstDimIndex = targetShape.getDimNum() - 1 - dim;
                int srcDimIndex = array.shape.getDimNum() - 1 - dim;

                if (targetShape.getDimension(dstDimIndex) == 1) {
                    // 如果目标维度为1，则索引始终为0
                    dstIndices[dstDimIndex] = 0;
                } else {
                    // 否则使用源索引
                    dstIndices[dstDimIndex] = srcIndices[srcDimIndex];
                }

                // 检查索引是否有效
                if (dstIndices[dstDimIndex] >= targetShape.getDimension(dstDimIndex)) {
                    valid = false;
                    break;
                }
            }

            // 如果索引有效，累加到目标位置
            if (valid) {
                int dstIndex = result.shape.getIndex(dstIndices);
                result.buffer[dstIndex] += array.buffer[i];
            }
        }

        return result;
    }

    // =============================================================================
    // 优化方法：新增增强API（保持向后兼容）
    // =============================================================================

    /**
     * 支持广播语义的 reshape，可将大小为 1 的维度扩展，如 [1,3] → [5,3]。
     *
     * @param array    源数组
     * @param newShape 新形状
     * @return 变形后的数组
     */
    public static NdArrayCpu broadcastReshape(NdArrayCpu array, Shape newShape) {
        int[] srcDims = array.shape.getShapeDims();
        int[] dstDims = newShape.getShapeDims();

        // 检查维度数量是否相同
        if (srcDims.length != dstDims.length) {
            throw new IllegalArgumentException(
                    String.format("维度数量必须相同: 源=%d, 目标=%d",
                            srcDims.length, dstDims.length));
        }

        boolean needsBroadcast = false;
        for (int i = 0; i < srcDims.length; i++) {
            if (srcDims[i] == 1 && dstDims[i] > 1) {
                // 需要广播
                needsBroadcast = true;
            } else if (srcDims[i] != dstDims[i]) {
                // 不兼容的形状
                throw new IllegalArgumentException(
                        String.format("维度[%d]不兼容: %d vs %d (只能从1扩展)",
                                i, srcDims[i], dstDims[i]));
            }
        }

        if (needsBroadcast) {
            // 使用broadcastTo实现扩展
            return array.broadcastTo(newShape);
        } else {
            // 常规reshape（零拷贝）
            return reshape(array, newShape);
        }
    }

    /**
     * 优化的 sumTo 实现，采用轴向求和策略，性能优于逐元素累加。
     *
     * @param array       源数组
     * @param targetShape 目标形状
     * @return 压缩结果数组
     */
    public static NdArrayCpu sumToOptimized(NdArrayCpu array, Shape targetShape) {
        int[] srcDims = array.shape.getShapeDims();
        int[] dstDims = targetShape.getShapeDims();

        // 检查形状兼容性
        if (array.shape.getDimNum() < targetShape.getDimNum()) {
            throw new IllegalArgumentException(
                    String.format("源形状维度(%d)不能小于目标形状维度(%d)",
                            array.shape.getDimNum(), targetShape.getDimNum()));
        }

        // 识别需要求和的轴
        java.util.List<Integer> sumAxes = new java.util.ArrayList<>();
        for (int i = 0; i < targetShape.getDimNum(); i++) {
            int srcDimIndex = array.shape.getDimNum() - targetShape.getDimNum() + i;
            int dstDimIndex = i;

            int srcDim = array.shape.getDimension(srcDimIndex);
            int dstDim = targetShape.getDimension(dstDimIndex);

            // sumTo规则验证
            if (srcDim != dstDim && dstDim != 1) {
                throw new IllegalArgumentException(
                        String.format("sumTo不兼容：源维度[%d]=%d，目标维度[%d]=%d",
                                srcDimIndex, srcDim, dstDimIndex, dstDim));
            }

            // 需要求和的维度
            if (dstDim == 1 && srcDim > 1) {
                sumAxes.add(srcDimIndex);
            }
        }

        // 如果没有需要求和的轴，直接reshape
        if (sumAxes.isEmpty()) {
            return (NdArrayCpu) array.reshape(targetShape);
        }

        // 逐轴求和（利用高效的sum(axis)实现）
        NdArrayCpu temp = array;
        // 按逆序求和（从后往前），避免维度索引变化
        java.util.Collections.sort(sumAxes, java.util.Collections.reverseOrder());

        for (int axis : sumAxes) {
            temp = (NdArrayCpu) temp.sum(axis);
        }

        // 最后reshape到目标形状
        return (NdArrayCpu) temp.reshape(targetShape);
    }
}

