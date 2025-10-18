package io.leavesfly.tinyai.ndarr;


import io.leavesfly.tinyai.ndarr.cpu.ShapeCpu;

public interface Shape {

    static Shape of(int row, int column) {
        return of(new int[]{row, column});
    }

    static Shape of(int... _dimension) {
        //先写死
        return new ShapeCpu(_dimension);
    }

    /**
     * 获取形状数组
     *
     * @return
     */
    int[] getShape();

    /**
     * 获取行数（仅适用于二维形状）
     *
     * @return 行数
     * @throws IllegalStateException 当形状不是二维时抛出异常
     */
    int getRow();

    /**
     * 获取列数（仅适用于二维形状）
     *
     * @return 列数
     * @throws IllegalStateException 当形状不是二维时抛出异常
     */
    int getColumn();

    /**
     * 判断是否是矩阵（二维形状）
     *
     * @return 如果是二维形状返回true，否则返回false
     */
    boolean isMatrix();

    /**
     * 判断是否是标量（零维形状）
     *
     * @return 如果是零维形状返回true，否则返回false
     */
    boolean isScalar();

    /**
     * 判断是否是向量（一维形状）
     *
     * @return 如果是一维形状返回true，否则返回false
     */
    boolean isVector();

    /**
     * 计算对应形状的N维数组的元素总数
     *
     * @return 元素总数
     */
    int size();

    /**
     * 根据多维索引计算一维数组中的位置
     *
     * @param indices 多维索引
     * @return 一维数组中的位置
     * @throws IllegalArgumentException  当索引维度与形状维度不匹配时抛出异常
     * @throws IndexOutOfBoundsException 当索引超出范围时抛出异常
     */
    int getIndex(int... indices);

    /**
     * 获取指定维度的大小
     *
     * @param dimIndex 维度索引
     * @return 指定维度的大小
     * @throws IndexOutOfBoundsException 当维度索引超出范围时抛出异常
     */
    int getDimension(int dimIndex);

    /**
     * 获取维度数量
     *
     * @return 维度数量
     */
    int getDimNum();

}
