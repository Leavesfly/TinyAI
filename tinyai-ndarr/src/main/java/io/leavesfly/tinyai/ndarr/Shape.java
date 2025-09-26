package io.leavesfly.tinyai.ndarr;

import java.io.Serializable;
import java.util.Arrays;

/**
 * 表示矩阵或向量等N维数组的形状
 * 
 * Shape类用于描述N维数组的维度信息，包含各维度的大小以及用于索引计算的乘数。
 * 该类是不可变的，一旦创建就不能修改。
 */
public class Shape implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * 各维度的大小
     */
    public final int[] dimension;

    /**
     * 各维度的乘数，用于快速索引计算
     * multipliers[i]表示第i维的一个单位在展开后的一维数组中占据的位置数
     */
    public final int[] multipliers;

    /**
     * 缓存的hashCode值，避免重复计算
     */
    private transient int hashCodeCache = 0;
    
    /**
     * 缓存的hashCode是否已计算
     */
    private transient boolean hashCodeComputed = false;

    /**
     * 构造二维形状
     * 
     * @param row 行数
     * @param column 列数
     */
    public Shape(int row, int column) {
        this(new int[]{row, column});
    }

    /**
     * 构造N维形状
     * 
     * @param _dimension 各维度的大小数组
     */
    public Shape(int... _dimension) {
        // 验证输入参数
        if (_dimension == null) {
            throw new IllegalArgumentException("维度数组不能为null");
        }
        
        // 复制维度数组以确保不可变性
        this.dimension = _dimension.clone();
        
        // 计算乘数数组
        this.multipliers = new int[_dimension.length];
        int accumulator = 1;
        for (int i = _dimension.length - 1; i >= 0; i--) {
//            // 检查维度是否为非负数
//            if (_dimension[i] < 0) {
//                throw new IllegalArgumentException("维度大小不能为负数: " + _dimension[i]);
//            }
            multipliers[i] = accumulator;
            accumulator *= _dimension[i];
        }
    }

    /**
     * 获取行数（仅适用于二维形状）
     * 
     * @return 行数
     * @throws IllegalStateException 当形状不是二维时抛出异常
     */
    public int getRow() {
        if (dimension.length != 2) {
            throw new IllegalStateException("只有二维形状才有行数，当前维度: " + dimension.length);
        }
        return dimension[0];
    }

    /**
     * 获取列数（仅适用于二维形状）
     * 
     * @return 列数
     * @throws IllegalStateException 当形状不是二维时抛出异常
     */
    public int getColumn() {
        if (dimension.length != 2) {
            throw new IllegalStateException("只有二维形状才有列数，当前维度: " + dimension.length);
        }
        return dimension[1];
    }

    /**
     * 判断是否是矩阵（二维形状）
     * 
     * @return 如果是二维形状返回true，否则返回false
     */
    public boolean isMatrix() {
        return dimension.length == 2;
    }
    
    /**
     * 判断是否是标量（零维形状）
     * 
     * @return 如果是零维形状返回true，否则返回false
     */
    public boolean isScalar() {
        return dimension.length == 0;
    }
    
    /**
     * 判断是否是向量（一维形状）
     * 
     * @return 如果是一维形状返回true，否则返回false
     */
    public boolean isVector() {
        return dimension.length == 1;
    }

    /**
     * 计算对应形状的N维数组的元素总数
     * 
     * @return 元素总数
     */
    public int size() {
        int size = 1;
        for (int dim : dimension) {
            size *= dim;
        }
        return size;
    }

    /**
     * 根据多维索引计算一维数组中的位置
     * 
     * @param indices 多维索引
     * @return 一维数组中的位置
     * @throws IllegalArgumentException 当索引维度与形状维度不匹配时抛出异常
     * @throws IndexOutOfBoundsException 当索引超出范围时抛出异常
     */
    public int getIndex(int... indices) {
        if (indices.length != dimension.length) {
            throw new IllegalArgumentException(
                String.format("索引维度(%d)与形状维度(%d)不匹配", indices.length, dimension.length));
        }
        
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            // 检查索引是否在有效范围内
            if (indices[i] < 0 || indices[i] >= dimension[i]) {
                throw new IndexOutOfBoundsException(
                    String.format("索引[%d]=%d超出范围[0,%d)", i, indices[i], dimension[i]));
            }
            index += indices[i] * multipliers[i];
        }
        return index;
    }
    
    /**
     * 获取指定维度的大小
     * 
     * @param dimIndex 维度索引
     * @return 指定维度的大小
     * @throws IndexOutOfBoundsException 当维度索引超出范围时抛出异常
     */
    public int getDimension(int dimIndex) {
        if (dimIndex < 0 || dimIndex >= dimension.length) {
            throw new IndexOutOfBoundsException(
                String.format("维度索引%d超出范围[0,%d)", dimIndex, dimension.length));
        }
        return dimension[dimIndex];
    }
    
    /**
     * 获取维度数量
     * 
     * @return 维度数量
     */
    public int getDimNum() {
        return dimension.length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return Arrays.equals(dimension, shape.dimension);
    }

    @Override
    public int hashCode() {
        // 使用缓存的hashCode以提高性能
        if (!hashCodeComputed) {
            // 使用Arrays.hashCode而不是Objects.hash以提高性能
            hashCodeCache = Arrays.hashCode(dimension);
            hashCodeComputed = true;
        }
        return hashCodeCache;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append('[');
        for (int i = 0; i < dimension.length; i++) {
            if (i > 0) {
                stringBuilder.append(',');
            }
            stringBuilder.append(dimension[i]);
        }
        stringBuilder.append(']');
        return stringBuilder.toString();
    }
}