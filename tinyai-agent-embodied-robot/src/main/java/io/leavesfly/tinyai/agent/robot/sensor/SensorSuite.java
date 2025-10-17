package io.leavesfly.tinyai.agent.robot.sensor;

import io.leavesfly.tinyai.agent.robot.env.CleaningEnvironment;
import io.leavesfly.tinyai.agent.robot.model.SensorType;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * 传感器组件
 * 
 * <p>集成和管理所有传感器。</p>
 * 
 * @author TinyAI Team
 */
public class SensorSuite {
    private Map<SensorType, Sensor> sensors;
    private CleaningEnvironment environment;
    
    public SensorSuite(CleaningEnvironment environment) {
        this.environment = environment;
        this.sensors = new HashMap<>();
        initializeSensors();
    }
    
    private void initializeSensors() {
        // 简化实现：使用默认传感器
        sensors.put(SensorType.CAMERA, new DefaultSensor(SensorType.CAMERA, 0.05));
        sensors.put(SensorType.LIDAR, new DefaultSensor(SensorType.LIDAR, 0.02));
        sensors.put(SensorType.CLIFF_SENSOR, new DefaultSensor(SensorType.CLIFF_SENSOR, 0.01));
        sensors.put(SensorType.BUMP_SENSOR, new DefaultSensor(SensorType.BUMP_SENSOR, 0.01));
        sensors.put(SensorType.DIRT_SENSOR, new DefaultSensor(SensorType.DIRT_SENSOR, 0.03));
        sensors.put(SensorType.ODOMETER, new DefaultSensor(SensorType.ODOMETER, 0.02));
    }
    
    public NdArray readAll() {
        List<Float> allData = new ArrayList<>();
        for (SensorType type : SensorType.values()) {
            if (sensors.containsKey(type)) {
                NdArray data = sensors.get(type).readData();
                for (int i = 0; i < data.getShape().size(); i++) {
                    allData.add(data.get(i));
                }
            }
        }
        
        float[] dataArray = new float[allData.size()];
        for (int i = 0; i < allData.size(); i++) {
            dataArray[i] = allData.get(i);
        }
        
        return NdArray.of(dataArray, Shape.of(allData.size()));
    }
    
    public NdArray readSensor(SensorType type) {
        Sensor sensor = sensors.get(type);
        return sensor != null ? sensor.readData() : NdArray.zeros(Shape.of(1));
    }
    
    public void calibrate() {
        sensors.values().forEach(Sensor::calibrate);
    }
    
    public Map<SensorType, Boolean> getSensorStatus() {
        Map<SensorType, Boolean> status = new HashMap<>();
        sensors.forEach((type, sensor) -> status.put(type, sensor.isReady()));
        return status;
    }
    
    /**
     * 默认传感器实现
     */
    private class DefaultSensor extends AbstractSensor {
        public DefaultSensor(SensorType type, double noiseLevel) {
            super(type, noiseLevel);
        }
        
        @Override
        public NdArray readData() {
            if (environment != null) {
                return environment.getSensorData(type);
            }
            return NdArray.zeros(Shape.of(type.getOutputDimension()));
        }
    }
}
