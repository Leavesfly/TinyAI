package io.leavesfly.tinyai.agent.robot.sensor;

import io.leavesfly.tinyai.agent.robot.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 传感器接口
 * 
 * <p>定义传感器的标准行为。</p>
 * 
 * @author TinyAI Team
 */
public interface Sensor {
    /**
     * 获取传感器类型
     * 
     * @return 传感器类型
     */
    SensorType getType();
    
    /**
     * 读取传感器数据
     * 
     * @return 传感器数据（NdArray格式）
     */
    NdArray readData();
    
    /**
     * 判断传感器是否就绪
     * 
     * @return 是否就绪
     */
    boolean isReady();
    
    /**
     * 重置传感器
     */
    void reset();
    
    /**
     * 获取噪声水平
     * 
     * @return 噪声水平（0-1）
     */
    double getNoiseLevel();
    
    /**
     * 校准传感器
     */
    void calibrate();
}
