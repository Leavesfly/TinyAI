package io.leavesfly.tinyai.ml.dataset;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;

/**
 * 一批数据
 * <p>
 * 该类表示机器学习中的一批训练或测试数据，包含输入数据和对应的标签数据。
 * 提供了将数据转换为Variable对象的方法，便于模型训练和推理。
 *
 * @author TinyDL
 * @version 1.0
 */
public class Batch {
    private NdArray[] x;
    private NdArray[] y;

    private int index = 0;

    // 缓存Variable实例以避免频繁创建对象
    private Variable cachedVariableX = null;
    private Variable cachedVariableY = null;

    /**
     * 构造函数
     *
     * @param x 输入数据数组
     * @param y 标签数据数组
     */
    public Batch(NdArray[] x, NdArray[] y) {
        this.x = x;
        this.y = y;
    }

    /**
     * 获取输入数据数组
     *
     * @return 输入数据数组
     */
    public NdArray[] getX() {
        return x;
    }

    /**
     * 设置输入数据数组
     *
     * @param x 输入数据数组
     */
    public void setX(NdArray[] x) {
        this.x = x;
        // 清除缓存
        this.cachedVariableX = null;
    }

    /**
     * 获取标签数据数组
     *
     * @return 标签数据数组
     */
    public NdArray[] getY() {
        return y;
    }

    /**
     * 设置标签数据数组
     *
     * @param y 标签数据数组
     */
    public void setY(NdArray[] y) {
        this.y = y;
        // 清除缓存
        this.cachedVariableY = null;
    }

    /**
     * 获取批次大小
     *
     * @return 批次大小
     */
    public int getSize() {
        return x.length;
    }

    /**
     * 将输入数据转换为Variable对象
     * 为了提高性能，该方法会缓存Variable实例
     *
     * @return 输入数据的Variable表示
     */
    public Variable toVariableX() {
        if (cachedVariableX == null) {
            cachedVariableX = new Variable(NdArrayUtil.merge(0, x));
        }
        return cachedVariableX;
    }

    /**
     * 将标签数据转换为Variable对象
     * 为了提高性能，该方法会缓存Variable实例
     *
     * @return 标签数据的Variable表示
     */
    public Variable toVariableY() {
        if (cachedVariableY == null) {
            cachedVariableY = new Variable(NdArrayUtil.merge(0, y));
        }
        return cachedVariableY;
    }

    /**
     * 获取下一对数据
     *
     * @return 数据对，如果已遍历完则返回null
     */
    public Pair<NdArray, NdArray> next() {
        if (index >= getSize()) {
            return null;
        }
        Pair<NdArray, NdArray> pair = new Pair<NdArray, NdArray>(x[index], y[index]);
        index++;
        return pair;
    }

    /**
     * 检查是否还有更多数据
     *
     * @return 如果还有数据返回true，否则返回false
     */
    public boolean hasNext() {
        return index < getSize();
    }

    /**
     * 重置遍历索引
     */
    public void resetIndex() {
        index = 0;
    }

    /**
     * 获取当前遍历索引
     *
     * @return 当前索引
     */
    public int getCurrentIndex() {
        return index;
    }

    /**
     * 键值对内部类
     *
     * @param <K> 键类型
     * @param <V> 值类型
     */
    public static class Pair<K, V> {
        public K key;
        public V value;

        /**
         * 构造函数
         *
         * @param key   键
         * @param value 值
         */
        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

}