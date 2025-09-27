package io.leavesfly.tinyai.func.matrix;


import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 索引获取函数
 * <p>
 * 根据指定的行列索引从输入数组中获取子数组。
 */
public class GetItem extends Function {
    private int[] rowSlices;
    private int[] colSlices;

    /**
     * 构造函数
     *
     * @param _rowSlices 行索引数组
     * @param _colSlices 列索引数组
     */
    public GetItem(int[] _rowSlices, int[] _colSlices) {
        this.rowSlices = _rowSlices;
        this.colSlices = _colSlices;
    }

    /**
     * 前向传播计算索引获取
     * <p>
     * 根据指定的行列索引从输入数组中获取子数组。
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return 索引获取后的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].getItem(rowSlices, colSlices);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * 对于索引获取操作，梯度计算通过在原始形状的零数组中将梯度值填充到指定位置。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray xGrad = NdArray.zeros(inputs[0].getValue().getShape()).addAt(rowSlices, colSlices, yGrad);
        return Collections.singletonList(xGrad);
    }

    /**
     * 获取所需输入参数个数
     * <p>
     * 索引获取函数需要一个输入参数。
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}
