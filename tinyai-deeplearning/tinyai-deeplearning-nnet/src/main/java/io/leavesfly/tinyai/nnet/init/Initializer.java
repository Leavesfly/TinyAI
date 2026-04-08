package io.leavesfly.tinyai.nnet.init;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 参数初始化器接口
 * <p>
 * 定义参数初始化的统一接口，实现类提供具体的初始化策略。
 * 初始化器操作已分配内存的NdArray，填充特定分布的数值。
 *
 * @author leavesfly
 * @version 2.0
 */
@FunctionalInterface
public interface Initializer {

    /**
     * 初始化给定的张量
     * <p>
     * 接收一个已分配内存的NdArray，按照特定策略填充数值。
     * 不改变张量的形状和设备位置。
     *
     * @param tensor 需要初始化的张量（已分配内存）
     * @throws IllegalArgumentException 当张量形状不满足初始化要求时抛出
     */
    void initialize(NdArray tensor);
}
