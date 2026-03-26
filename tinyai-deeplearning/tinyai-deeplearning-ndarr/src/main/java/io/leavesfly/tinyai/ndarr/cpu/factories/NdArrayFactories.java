package io.leavesfly.tinyai.ndarr.cpu.factories;

import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

import java.util.Arrays;
import java.util.Random;

/**
 * NdArray 工厂类，提供各类创建 NdArray 的静态工厂方法。
 */
public class NdArrayFactories {

    /**
     * 创建指定形状的全零数组
     *
     * @param shape 数组形状
     * @return 全零数组
     */
    public static NdArrayCpu zeros(Shape shape) {
        if (shape == null) {
            throw new IllegalArgumentException("shape不能为null");
        }
        return new NdArrayCpu(shape);
    }

    /**
     * 创建指定形状的全一数组
     *
     * @param shape 数组形状
     * @return 全一数组
     */
    public static NdArrayCpu ones(Shape shape) {
        if (shape == null) {
            throw new IllegalArgumentException("shape不能为null");
        }
        NdArrayCpu result = new NdArrayCpu(shape);
        fillAll(result, 1.0f);
        return result;
    }

    /**
     * 创建指定形状的单位矩阵（对角矩阵）
     *
     * @param shape 矩阵形状（必须为方形矩阵）
     * @return 单位矩阵
     * @throws IllegalArgumentException 当形状不是矩阵或不是方形矩阵时抛出
     */
    public static NdArrayCpu eye(Shape shape) {
        if (shape == null) {
            throw new IllegalArgumentException("shape不能为null");
        }
        if (shape.getDimNum() >= 2) {
            NdArrayCpu result = new NdArrayCpu(shape);
            int lastDimSize = shape.getDimension(shape.getDimNum() - 1);
            int secondLastDimSize = shape.getDimension(shape.getDimNum() - 2);
            int minDim = Math.min(lastDimSize, secondLastDimSize);

            // 计算批次大小
            int batchSize = shape.size() / (lastDimSize * secondLastDimSize);

            for (int batch = 0; batch < batchSize; batch++) {
                for (int i = 0; i < minDim; i++) {
                    // 计算多维索引（简化处理）
                    int index = batch * (lastDimSize * secondLastDimSize) + i * lastDimSize + i;
                    result.buffer[index] = 1.0f;
                }
            }
            return result;
        }

        throw new IllegalArgumentException("操作需要至少二维数组");
    }

    /**
     * 创建指定形状和值的数组
     *
     * @param shape 数组形状
     * @param value 填充值
     * @return 指定值填充的数组
     */
    public static NdArrayCpu like(Shape shape, Number value) {
        if (shape == null) {
            throw new IllegalArgumentException("shape不能为null");
        }
        NdArrayCpu result = new NdArrayCpu(shape);
        fillAll(result, value.floatValue());
        return result;
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组
     *
     * @param shape 数组形状
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape) {
        return likeRandomN(shape, 0);
    }

    /**
     * 创建标准正态分布（均值为0，标准差为1）的随机数组（可指定随机种子）
     *
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 标准正态分布随机数组
     */
    public static NdArrayCpu likeRandomN(Shape shape, long seed) {
        NdArrayCpu result = new NdArrayCpu(shape);
        Random random = seed == 0 ? new Random() : new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = (float) random.nextGaussian();
        }
        return result;
    }

    /**
     * 创建指定范围内的均匀分布随机数组
     *
     * @param min   最小值（包含）
     * @param max   最大值（包含）
     * @param shape 数组形状
     * @return 均匀分布随机数组
     */
    public static NdArrayCpu likeRandom(float min, float max, Shape shape) {
        return likeRandom(min, max, shape, 0);
    }

    /**
     * 创建指定范围内的均匀分布随机数组（可指定随机种子）
     *
     * @param min   最小值（包含）
     * @param max   最大值（包含）
     * @param shape 数组形状
     * @param seed  随机种子，0表示使用默认种子
     * @return 均匀分布随机数组
     */
    public static NdArrayCpu likeRandom(float min, float max, Shape shape, long seed) {
        NdArrayCpu result = new NdArrayCpu(shape);
        Random random = seed == 0 ? new Random() : new Random(seed);
        for (int i = 0; i < result.buffer.length; i++) {
            result.buffer[i] = random.nextFloat() * (max - min) + min;
        }
        return result;
    }

    /**
     * 创建线性空间数组（等间距排序数组）
     *
     * @param min 起始值
     * @param max 结束值
     * @param num 元素数量
     * @return 线性空间数组
     * @throws IllegalArgumentException 当数量小于等于0时抛出
     */
    public static NdArrayCpu linSpace(float min, float max, int num) {
        if (num <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }
        NdArrayCpu result = new NdArrayCpu(ShapeCpu.of(1, num));
        if (num == 1) {
            result.buffer[0] = min;
            return result;
        }
        float step = (max - min) / (num - 1);
        for (int i = 0; i < num; i++) {
            result.buffer[i] = min + step * i;
        }
        return result;
    }

    /**
     * 用指定值填充整个数组
     *
     * @param array 数组
     * @param value 填充值
     */
    private static void fillAll(NdArrayCpu array, float value) {
        Arrays.fill(array.buffer, value);
    }
}

