package io.leavesfly.tinyai.ndarr;

import java.util.Arrays;

/**
 * N维数组类 - TinyAI项目的核心数据结构
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class NDArray {
    
    /** 数组数据 */
    private double[] data;
    
    /** 数组形状 */
    private int[] shape;
    
    /** 数组总元素个数 */
    private int size;
    
    /**
     * 构造函数 - 根据形状创建N维数组
     * 
     * @param shape 数组形状，例如 [2, 3, 4] 表示2x3x4的三维数组
     */
    public NDArray(int... shape) {
        this.shape = shape.clone();
        this.size = calculateSize(shape);
        this.data = new double[size];
    }
    
    /**
     * 构造函数 - 根据数据和形状创建N维数组
     * 
     * @param data 一维数据数组
     * @param shape 数组形状
     */
    public NDArray(double[] data, int... shape) {
        this.shape = shape.clone();
        this.size = calculateSize(shape);
        if (data.length != size) {
            throw new IllegalArgumentException("数据长度与形状不匹配");
        }
        this.data = data.clone();
    }
    
    /**
     * 计算数组总大小
     * 
     * @param shape 数组形状
     * @return 总元素个数
     */
    private int calculateSize(int[] shape) {
        int totalSize = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("数组维度必须大于0");
            }
            totalSize *= dim;
        }
        return totalSize;
    }
    
    /**
     * 设置指定位置的值
     * 
     * @param value 要设置的值
     * @param indices 多维索引
     */
    public void set(double value, int... indices) {
        int index = getLinearIndex(indices);
        data[index] = value;
    }
    
    /**
     * 获取指定位置的值
     * 
     * @param indices 多维索引
     * @return 对应位置的值
     */
    public double get(int... indices) {
        int index = getLinearIndex(indices);
        return data[index];
    }
    
    /**
     * 将多维索引转换为一维索引
     * 
     * @param indices 多维索引
     * @return 一维索引
     */
    private int getLinearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("索引维度不匹配");
        }
        
        int linearIndex = 0;
        int multiplier = 1;
        
        for (int i = shape.length - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("索引超出范围");
            }
            linearIndex += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        
        return linearIndex;
    }
    
    /**
     * 用指定值填充整个数组
     * 
     * @param value 填充值
     * @return 当前数组对象，支持链式调用
     */
    public NDArray fill(double value) {
        Arrays.fill(data, value);
        return this;
    }
    
    /**
     * 数组加法
     * 
     * @param other 另一个数组
     * @return 新的结果数组
     */
    public NDArray add(NDArray other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("数组形状不匹配，无法进行加法运算");
        }
        
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.data[i] + other.data[i];
        }
        
        return new NDArray(result, shape);
    }
    
    /**
     * 数组乘法（标量）
     * 
     * @param scalar 标量值
     * @return 新的结果数组
     */
    public NDArray multiply(double scalar) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = this.data[i] * scalar;
        }
        
        return new NDArray(result, shape);
    }
    
    /**
     * 获取数组形状
     * 
     * @return 数组形状的副本
     */
    public int[] getShape() {
        return shape.clone();
    }
    
    /**
     * 获取数组大小
     * 
     * @return 总元素个数
     */
    public int getSize() {
        return size;
    }
    
    /**
     * 获取数组维度
     * 
     * @return 数组维度数
     */
    public int getDimensions() {
        return shape.length;
    }
    
    @Override
    public String toString() {
        return "NDArray{" +
                "shape=" + Arrays.toString(shape) +
                ", size=" + size +
                ", data=" + Arrays.toString(Arrays.copyOf(data, Math.min(10, data.length))) +
                (data.length > 10 ? "..." : "") +
                '}';
    }
}