package io.leavesfly.tinyai.agent.embodied.sensor;

import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 抽象传感器基类
 * 提供传感器的通用实现
 *
 * @author TinyAI Team
 */
public abstract class AbstractSensor implements Sensor {
    protected DrivingEnvironment environment;
    protected SensorType type;
    protected double noiseLevel;
    protected boolean ready;

    public AbstractSensor(DrivingEnvironment environment, SensorType type) {
        this.environment = environment;
        this.type = type;
        this.noiseLevel = 0.01;  // 默认1%噪声
        this.ready = true;
    }

    @Override
    public SensorType getType() {
        return type;
    }

    @Override
    public String getName() {
        return type.getName();
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    @Override
    public void reset() {
        ready = true;
    }

    @Override
    public double getNoiseLevel() {
        return noiseLevel;
    }

    @Override
    public void setNoiseLevel(double noiseLevel) {
        this.noiseLevel = Math.max(0.0, Math.min(1.0, noiseLevel));
    }

    /**
     * 添加噪声到数据
     */
    protected NdArray addNoise(NdArray data) {
        if (noiseLevel < 1e-6) {
            return data;
        }
        
        // 添加高斯噪声
        NdArray noise = NdArray.likeRandomN(data.getShape());
        noise = noise.mul(NdArray.of((float)noiseLevel));
        return data.add(noise);
    }
}

/**
 * 相机传感器
 */
class CameraSensor extends AbstractSensor {
    public CameraSensor(DrivingEnvironment environment) {
        super(environment, SensorType.CAMERA);
    }

    @Override
    public NdArray readData() {
        // 从环境获取原始数据并添加噪声
        NdArray rawData = environment.getSensorData(SensorType.CAMERA);
        return addNoise(rawData);
    }
}

/**
 * 激光雷达传感器
 */
class LidarSensor extends AbstractSensor {
    public LidarSensor(DrivingEnvironment environment) {
        super(environment, SensorType.LIDAR);
    }

    @Override
    public NdArray readData() {
        NdArray rawData = environment.getSensorData(SensorType.LIDAR);
        return addNoise(rawData);
    }
}

/**
 * IMU惯性测量单元
 */
class IMUSensor extends AbstractSensor {
    public IMUSensor(DrivingEnvironment environment) {
        super(environment, SensorType.IMU);
        this.noiseLevel = 0.05;  // IMU噪声较大
    }

    @Override
    public NdArray readData() {
        NdArray rawData = environment.getSensorData(SensorType.IMU);
        return addNoise(rawData);
    }
}

/**
 * GPS定位传感器
 */
class GPSSensor extends AbstractSensor {
    public GPSSensor(DrivingEnvironment environment) {
        super(environment, SensorType.GPS);
        this.noiseLevel = 0.02;
    }

    @Override
    public NdArray readData() {
        NdArray rawData = environment.getSensorData(SensorType.GPS);
        return addNoise(rawData);
    }
}

/**
 * 速度传感器
 */
class SpeedometerSensor extends AbstractSensor {
    public SpeedometerSensor(DrivingEnvironment environment) {
        super(environment, SensorType.SPEEDOMETER);
        this.noiseLevel = 0.005;  // 速度传感器精度高
    }

    @Override
    public NdArray readData() {
        NdArray rawData = environment.getSensorData(SensorType.SPEEDOMETER);
        return addNoise(rawData);
    }
}
