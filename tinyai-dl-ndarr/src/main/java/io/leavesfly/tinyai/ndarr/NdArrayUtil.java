package io.leavesfly.tinyai.ndarr;


/**
 * NdArray工具类，提供数组操作的辅助方法
 */
public class NdArrayUtil {

    /**
     * 按照指定轴对多个NdArray进行合并
     *
     * @param axis     合并的轴向，0表示按第一个维度合并，1表示按第二个维度合并，以此类推
     * @param ndArrays 需要合并的NdArray数组
     * @return 合并后的NdArray
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {
        // 验证输入参数
        if (ndArrays == null || ndArrays.length == 0) {
            throw new IllegalArgumentException("至少需要一个NdArray进行合并");
        }

        // 如果只有一个数组，直接返回副本
        if (ndArrays.length == 1) {
            NdArray original = ndArrays[0];
            if (original.getShape().isMatrix()) {
                return NdArray.of(original.getMatrix());
            } else {
                // 对于其他维度，创建新的数组
                NdArray result = NdArray.of(original.getShape());
                // 复制数据
                for (int i = 0; i < original.getShape().size(); i++) {
                    copyDataByIndex(original, result, i);
                }
                return result;
            }
        }

        // 获取第一个数组作为参考
        NdArray first = ndArrays[0];
        Shape firstShape = first.getShape();

        // 验证轴参数的有效性
        if (axis < 0 || axis >= firstShape.getDimNum()) {
            throw new RuntimeException("axis参数超出数组维度范围: " + axis);
        }

        // 验证所有数组的形状兼容性（除了指定的合并轴）
        validateMergeCompatibility(ndArrays, axis);

        // 计算合并后的新形状
        Shape mergedShape = calculateMergedShape(ndArrays, axis);

        // 创建结果数组
        NdArray result = NdArray.of(mergedShape);

        // 执行合并操作
        if (firstShape.isMatrix()) {
            mergeMatrices(result, ndArrays, axis);
        } else {
            mergeMultiDimensional(result, ndArrays, axis);
        }

        return result;
    }


    /**
     * 生成从0开始的连续整数序列
     *
     * @param size 序列长度
     * @return 连续整数数组
     */
    public static int[] getSeq(int size) {
        int[] seq = new int[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }

    /**
     * 复制单个数据元素（按索引）
     */
    private static void copyDataByIndex(NdArray source, NdArray target, int index) {
        // 通过多维索引复制数据
        int[] sourceIndices = getIndicesFromFlatIndex(source.getShape(), index);
        int[] targetIndices = getIndicesFromFlatIndex(target.getShape(), index);
        float value = source.get(sourceIndices);
        target.set(value, targetIndices);
    }

    /**
     * 从平坦索引获取多维索引
     */
    private static int[] getIndicesFromFlatIndex(Shape shape, int flatIndex) {
        int[] indices = new int[shape.getDimNum()];
        int remaining = flatIndex;

        for (int dim = shape.getDimNum() - 1; dim >= 0; dim--) {
            int dimSize = shape.getDimension(dim);
            indices[dim] = remaining % dimSize;
            remaining /= dimSize;
        }

        return indices;
    }

    /**
     * 验证数组合并的兼容性
     */
    private static void validateMergeCompatibility(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();

        for (int i = 1; i < ndArrays.length; i++) {
            Shape currentShape = ndArrays[i].getShape();

            // 检查维度数是否相同
            if (currentShape.getDimNum() != firstShape.getDimNum()) {
                throw new IllegalArgumentException(
                        String.format("数组%d的维度数(%d)与第一个数组的维度数(%d)不匹配",
                                i, currentShape.getDimNum(), firstShape.getDimNum()));
            }

            // 检查除合并轴外的其他维度是否相同
            for (int dim = 0; dim < firstShape.getDimNum(); dim++) {
                if (dim != axis) {
                    if (currentShape.getDimension(dim) != firstShape.getDimension(dim)) {
                        throw new IllegalArgumentException(
                                String.format("数组%d在维度%d上的大小(%d)与第一个数组(%d)不匹配",
                                        i, dim, currentShape.getDimension(dim), firstShape.getDimension(dim)));
                    }
                }
            }
        }
    }

    /**
     * 计算合并后的形状
     */
    private static Shape calculateMergedShape(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();
        int[] newDimensions = new int[firstShape.getDimNum()];

        // 复制所有维度
        for (int i = 0; i < firstShape.getDimNum(); i++) {
            newDimensions[i] = firstShape.getDimension(i);
        }

        // 计算合并轴上的总大小
        int totalSize = 0;
        for (NdArray array : ndArrays) {
            totalSize += array.getShape().getDimension(axis);
        }
        newDimensions[axis] = totalSize;

        return Shape.of(newDimensions);
    }

    /**
     * 合并矩阵（二维数组）
     */
    private static void mergeMatrices(NdArray result, NdArray[] ndArrays, int axis) {
        if (axis == 0) {
            // 按行合并
            int currentRow = 0;
            for (NdArray array : ndArrays) {
                float[][] matrix = array.getMatrix();
                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < matrix[i].length; j++) {
                        result.set(matrix[i][j], currentRow + i, j);
                    }
                }
                currentRow += matrix.length;
            }
        } else if (axis == 1) {
            // 按列合并
            int currentCol = 0;
            for (NdArray array : ndArrays) {
                float[][] matrix = array.getMatrix();
                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < matrix[i].length; j++) {
                        result.set(matrix[i][j], i, currentCol + j);
                    }
                }
                currentCol += matrix[0].length;
            }
        }
    }

    /**
     * 合并多维数组
     */
    private static void mergeMultiDimensional(NdArray result, NdArray[] ndArrays, int axis) {
        int[] offsetInAxis = new int[1]; // 使用数组来实现引用传递
        offsetInAxis[0] = 0;

        for (NdArray array : ndArrays) {
            copyArrayToResult(array, result, axis, offsetInAxis);
            offsetInAxis[0] += array.getShape().getDimension(axis);
        }
    }

    /**
     * 将数组复制到结果数组中
     */
    private static void copyArrayToResult(NdArray source, NdArray target, int axis, int[] offsetInAxis) {
        Shape sourceShape = source.getShape();
        copyRecursive(source, target, sourceShape, new int[sourceShape.getDimNum()], 0, axis, offsetInAxis[0]);
    }

    /**
     * 递归复制多维数组
     */
    private static void copyRecursive(NdArray source, NdArray target, Shape sourceShape,
                                      int[] indices, int currentDim, int mergeAxis, int offset) {
        if (currentDim == sourceShape.getDimNum()) {
            // 到达叶子节点，复制数据
            float value = source.get(indices);
            int[] targetIndices = indices.clone();
            targetIndices[mergeAxis] += offset;
            target.set(value, targetIndices);
            return;
        }

        // 遍历当前维度
        for (int i = 0; i < sourceShape.getDimension(currentDim); i++) {
            indices[currentDim] = i;
            copyRecursive(source, target, sourceShape, indices, currentDim + 1, mergeAxis, offset);
        }
    }


    /**
     * 将浮点数组转换为整型数组
     *
     * @param src 浮点数组
     * @return 整型数组
     */
    public static int[] toInt(float[] src) {
        int[] res = new int[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = (int) src[i];
        }
        return res;
    }
}