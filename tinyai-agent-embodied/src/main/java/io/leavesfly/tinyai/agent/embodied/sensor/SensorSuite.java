package io.leavesfly.tinyai.agent.embodied.sensor;

import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.*;

/**
 * 传感器组件集合
 * 管理多个传感器并提供统一的数据访问接口
 *
 * @author TinyAI Team
 */
public class SensorSuite {
    private Map<SensorType, Sensor> sensors;
    private DrivingEnvironment environment;
    private boolean initialized;

    public SensorSuite(DrivingEnvironment environment) {
        this.environment = environment;
        this.sensors = new HashMap<>();
        this.initialized = false;
    }

    /**
     * 初始化所有传感器
     */
    public void initialize() {
        // 添加默认传感器
        addSensor(new CameraSensor(environment));
        addSensor(new LidarSensor(environment));
        addSensor(new IMUSensor(environment));
        addSensor(new GPSSensor(environment));
        addSensor(new SpeedometerSensor(environment));
        
        initialized = true;
    }

    /**
     * 添加传感器
     */
    public void addSensor(Sensor sensor) {
        sensors.put(sensor.getType(), sensor);
    }

    /**
     * 移除传感器
     */
    public void removeSensor(SensorType type) {
        sensors.remove(type);
    }

    /**
     * 获取指定类型的传感器
     */
    public Sensor getSensor(SensorType type) {
        return sensors.get(type);
    }

    /**
     * 读取所有传感器数据
     */
    public Map<SensorType, NdArray> readAllSensors() {
        Map<SensorType, NdArray> data = new HashMap<>();
        for (Map.Entry<SensorType, Sensor> entry : sensors.entrySet()) {
            if (entry.getValue().isReady()) {
                data.put(entry.getKey(), entry.getValue().readData());
            }
        }
        return data;
    }

    /**
     * 读取指定传感器数据
     */
    public NdArray readSensor(SensorType type) {
        Sensor sensor = sensors.get(type);
        if (sensor != null && sensor.isReady()) {
            return sensor.readData();
        }
        return null;
    }

    /**
     * 重置所有传感器
     */
    public void resetAll() {
        for (Sensor sensor : sensors.values()) {
            sensor.reset();
        }
    }

    /**
     * 获取传感器数量
     */
    public int getSensorCount() {
        return sensors.size();
    }

    /**
     * 检查所有传感器是否就绪
     */
    public boolean allSensorsReady() {
        for (Sensor sensor : sensors.values()) {
            if (!sensor.isReady()) {
                return false;
            }
        }
        return true;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public Set<SensorType> getAvailableSensorTypes() {
        return sensors.keySet();
    }
}
