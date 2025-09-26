package io.leavesfly.tinyai.ndarr;

import java.util.Arrays;

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
        //todo
        return null;
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
}