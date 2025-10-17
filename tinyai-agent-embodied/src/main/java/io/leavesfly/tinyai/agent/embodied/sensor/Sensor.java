package io.leavesfly.tinyai.agent.embodied.sensor;

import io.leavesfly.tinyai.agent.embodied.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 传感器接口
 * 定义所有传感器的统一规范
 *
 * @author TinyAI Team
 */
public interface Sensor {
    /**
     * 获取传感器类型
     */
    SensorType getType();

    /**
     * 读取传感器数据
     * @return 传感器输出的原始数据
     */
    NdArray readData();

    /**
     * 获取传感器名称
     */
    String getName();

    /**
     * 传感器是否就绪
     */
    boolean isReady();

    /**
     * 重置传感器
     */
    void reset();

    /**
     * 获取传感器精度/噪声水平
     */
    double getNoiseLevel();

    /**
     * 设置传感器噪声水平
     */
    void setNoiseLevel(double noiseLevel);
}
