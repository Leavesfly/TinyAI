package io.leavesfly.tinyai.agent.robot.sensor;

import io.leavesfly.tinyai.agent.robot.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 抽象传感器基类
 * 
 * <p>提供传感器的通用实现。</p>
 * 
 * @author TinyAI Team
 */
public abstract class AbstractSensor implements Sensor {
    protected SensorType type;
    protected double noiseLevel;
    protected boolean ready;
    
    public AbstractSensor(SensorType type, double noiseLevel) {
        this.type = type;
        this.noiseLevel = noiseLevel;
        this.ready = true;
    }
    
    @Override
    public SensorType getType() {
        return type;
    }
    
    @Override
    public boolean isReady() {
        return ready;
    }
    
    @Override
    public double getNoiseLevel() {
        return noiseLevel;
    }
    
    @Override
    public void reset() {
        ready = true;
    }
    
    @Override
    public void calibrate() {
        // 默认实现：重置传感器
        reset();
    }
    
    /**
     * 添加高斯噪声
     * 
     * @param data 原始数据
     * @return 添加噪声后的数据
     */
    protected NdArray addNoise(NdArray data) {
        if (noiseLevel <= 0) {
            return data;
        }
        
        NdArray noise = NdArray.likeRandomN(data.getShape());
        NdArray scale = NdArray.of((float) noiseLevel);
        return data.add(noise.mul(scale));
    }
}
