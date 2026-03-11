package io.leavesfly.tinyai.ndarr;

import java.util.Arrays;

/**
 * NdArray工具类，提供数组操作的辅助方法。
 * 只依赖NdArray接口定义的方法，不依赖具体实现，确保代码的可移植性。
 */
public class NdArrayUtil {

    /**
     * 按照指定轴对多个NdArray进行合并。
     *
     * @param axis     合并轴，0表示按第一个维度合并，1表示按第二个维度合并，以此类推
     * @param ndArrays 需要合并的NdArray数组
     * @return 合并后的NdArray
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {
        if (ndArrays == null || ndArrays.length == 0) {
            throw new IllegalArgumentException("至少需要一个NdArray进行合并");
        }
        if (ndArrays.length == 1) {
            return copyNdArray(ndArrays[0]);
        }

        Shape firstShape = ndArrays[0].getShape();
        if (axis < 0 || axis >= firstShape.getDimNum()) {
            throw new RuntimeException("axis参数超出数组维度范围: " + axis);
        }

        validateMergeCompatibility(ndArrays, axis);

        NdArray result = NdArray.of(calculateMergedShape(ndArrays, axis));
        mergeArrays(result, ndArrays, axis);
        return result;
    }

    /**
     * 复制NdArray（不依赖具体实现）。
     *
     * @param source 源数组
     * @return 复制后的新数组
     */
    private static NdArray copyNdArray(NdArray source) {
        float[] sourceData = source.getArray();
        return NdArray.of(Arrays.copyOf(sourceData, sourceData.length), source.getShape());
    }

    /**
     * 验证待合并数组之间的形状兼容性（除合并轴外其余维度须一致）。
     *
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     * @throws IllegalArgumentException 当数组不兼容时抛出
     */
    private static void validateMergeCompatibility(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();
        int dimNum = firstShape.getDimNum();

        for (int i = 1; i < ndArrays.length; i++) {
            Shape shape = ndArrays[i].getShape();
            if (shape.getDimNum() != dimNum) {
                throw new IllegalArgumentException(
                        String.format("数组%d的维度数(%d)与第一个数组的维度数(%d)不匹配",
                                i, shape.getDimNum(), dimNum));
            }
            for (int dim = 0; dim < dimNum; dim++) {
                if (dim != axis && shape.getDimension(dim) != firstShape.getDimension(dim)) {
                    throw new IllegalArgumentException(
                            String.format("数组%d在维度%d上的大小(%d)与第一个数组(%d)不匹配",
                                    i, dim, shape.getDimension(dim), firstShape.getDimension(dim)));
                }
            }
        }
    }

    /**
     * 计算合并后的形状：合并轴的大小为各数组在该轴上的大小之和，其余维度不变。
     *
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     * @return 合并后的形状
     */
    private static Shape calculateMergedShape(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();
        int[] dims = new int[firstShape.getDimNum()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = firstShape.getDimension(i);
        }
        int totalAxisSize = 0;
        for (NdArray array : ndArrays) {
            totalAxisSize += array.getShape().getDimension(axis);
        }
        dims[axis] = totalAxisSize;
        return Shape.of(dims);
    }

    /**
     * 将各源数组按指定轴拼接写入结果数组。
     * 合并策略：
     *  - axis=0：各数组数据连续存储，逐块顺序复制
     *  - axis>0：按"外层切片 × 源块"的方式逐块复制
     *
     * @param result   结果数组
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     */
    private static void mergeArrays(NdArray result, NdArray[] ndArrays, int axis) {
        Shape resultShape = result.getShape();
        float[] resultData = result.getArray();
        int dimNum = resultShape.getDimNum();

        // 合并轴之前所有维度的乘积（外层切片数）
        int outerSize = 1;
        for (int i = 0; i < axis; i++) {
            outerSize *= resultShape.getDimension(i);
        }

        // 合并轴之后所有维度的乘积（单个轴单元对应的元素数）
        int innerSize = 1;
        for (int i = axis + 1; i < dimNum; i++) {
            innerSize *= resultShape.getDimension(i);
        }

        int resultAxisSize = resultShape.getDimension(axis);
        int resultAxisOffset = 0;

        for (NdArray src : ndArrays) {
            float[] srcData = src.getArray();
            int srcAxisSize = src.getShape().getDimension(axis);
            int srcBlockSize = srcAxisSize * innerSize;

            for (int outer = 0; outer < outerSize; outer++) {
                int srcStart = outer * srcBlockSize;
                int dstStart = outer * (resultAxisSize * innerSize) + resultAxisOffset * innerSize;
                System.arraycopy(srcData, srcStart, resultData, dstStart, srcBlockSize);
            }
            resultAxisOffset += srcAxisSize;
        }
    }

    /**
     * 将浮点数组转换为整型数组。
     *
     * @param src 浮点数组
     * @return 整型数组，若入参为null则返回null
     */
    public static int[] toInt(float[] src) {
        if (src == null) {
            return null;
        }
        int[] res = new int[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = (int) src[i];
        }
        return res;
    }

    /**
     * 生成从0开始的连续整数序列 [0, 1, ..., size-1]。
     *
     * @param size 序列长度
     * @return 连续整数数组
     * @throws IllegalArgumentException 当size小于0时抛出
     */
    public static int[] getSeq(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("序列长度不能为负数: " + size);
        }
        int[] seq = new int[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }
}
