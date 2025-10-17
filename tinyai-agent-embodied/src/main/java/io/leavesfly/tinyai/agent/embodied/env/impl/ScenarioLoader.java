package io.leavesfly.tinyai.agent.embodied.env.impl;

import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.model.ScenarioType;

import java.util.HashMap;
import java.util.Map;

/**
 * 场景加载器
 * 负责加载和管理不同的驾驶场景配置
 *
 * @author TinyAI Team
 */
public class ScenarioLoader {
    private Map<ScenarioType, EnvironmentConfig> scenarioRegistry;

    public ScenarioLoader() {
        this.scenarioRegistry = new HashMap<>();
        registerDefaultScenarios();
    }

    /**
     * 注册默认场景
     */
    private void registerDefaultScenarios() {
        // 测试场景
        registerScenario(ScenarioType.TEST, EnvironmentConfig.createTestConfig());
        
        // 高速公路场景
        registerScenario(ScenarioType.HIGHWAY, EnvironmentConfig.createHighwayConfig());
        
        // 城市道路场景
        registerScenario(ScenarioType.URBAN, EnvironmentConfig.createUrbanConfig());
        
        // 乡村道路场景
        EnvironmentConfig ruralConfig = new EnvironmentConfig();
        ruralConfig.setScenarioType(ScenarioType.RURAL);
        ruralConfig.setLaneCount(2);
        ruralConfig.setVehicleDensity(10);
        ruralConfig.setSpeedLimit(80.0);
        ruralConfig.setCurvatureRadius(200.0);
        registerScenario(ScenarioType.RURAL, ruralConfig);
        
        // 停车场场景
        EnvironmentConfig parkingConfig = new EnvironmentConfig();
        parkingConfig.setScenarioType(ScenarioType.PARKING_LOT);
        parkingConfig.setLaneCount(1);
        parkingConfig.setVehicleDensity(50);
        parkingConfig.setSpeedLimit(20.0);
        parkingConfig.setRoadLength(100.0);
        parkingConfig.setMaxSteps(500);
        registerScenario(ScenarioType.PARKING_LOT, parkingConfig);
        
        // 路口场景
        EnvironmentConfig intersectionConfig = new EnvironmentConfig();
        intersectionConfig.setScenarioType(ScenarioType.INTERSECTION);
        intersectionConfig.setLaneCount(3);
        intersectionConfig.setVehicleDensity(30);
        intersectionConfig.setSpeedLimit(50.0);
        intersectionConfig.setCurvatureRadius(50.0);
        registerScenario(ScenarioType.INTERSECTION, intersectionConfig);
    }

    /**
     * 注册场景
     */
    public void registerScenario(ScenarioType type, EnvironmentConfig config) {
        scenarioRegistry.put(type, config);
    }

    /**
     * 加载场景配置
     */
    public EnvironmentConfig loadScenario(ScenarioType type) {
        EnvironmentConfig config = scenarioRegistry.get(type);
        if (config == null) {
            throw new IllegalArgumentException("Unknown scenario type: " + type);
        }
        return config;
    }

    /**
     * 创建自定义场景
     */
    public EnvironmentConfig createCustomScenario(ScenarioType baseType,
                                                  Map<String, Object> customParams) {
        EnvironmentConfig baseConfig = loadScenario(baseType);
        EnvironmentConfig customConfig = cloneConfig(baseConfig);
        
        // 应用自定义参数
        applyCustomParams(customConfig, customParams);
        
        return customConfig;
    }

    /**
     * 克隆配置
     */
    private EnvironmentConfig cloneConfig(EnvironmentConfig source) {
        EnvironmentConfig target = new EnvironmentConfig();
        target.setScenarioType(source.getScenarioType());
        target.setLaneCount(source.getLaneCount());
        target.setLaneWidth(source.getLaneWidth());
        target.setRoadLength(source.getRoadLength());
        target.setCurvatureRadius(source.getCurvatureRadius());
        target.setVehicleDensity(source.getVehicleDensity());
        target.setSpeedLimit(source.getSpeedLimit());
        target.setTargetSpeed(source.getTargetSpeed());
        target.setVisibility(source.getVisibility());
        target.setFrictionCoeff(source.getFrictionCoeff());
        target.setTimeStep(source.getTimeStep());
        target.setMaxSteps(source.getMaxSteps());
        target.setRewardSpeedWeight(source.getRewardSpeedWeight());
        target.setRewardLaneWeight(source.getRewardLaneWeight());
        target.setRewardCollisionWeight(source.getRewardCollisionWeight());
        target.setRewardComfortWeight(source.getRewardComfortWeight());
        return target;
    }

    /**
     * 应用自定义参数
     */
    private void applyCustomParams(EnvironmentConfig config, Map<String, Object> params) {
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            
            switch (key) {
                case "laneCount":
                    config.setLaneCount((Integer) value);
                    break;
                case "laneWidth":
                    config.setLaneWidth((Double) value);
                    break;
                case "roadLength":
                    config.setRoadLength((Double) value);
                    break;
                case "curvatureRadius":
                    config.setCurvatureRadius((Double) value);
                    break;
                case "vehicleDensity":
                    config.setVehicleDensity((Integer) value);
                    break;
                case "speedLimit":
                    config.setSpeedLimit((Double) value);
                    break;
                case "targetSpeed":
                    config.setTargetSpeed((Double) value);
                    break;
                case "visibility":
                    config.setVisibility((Double) value);
                    break;
                case "frictionCoeff":
                    config.setFrictionCoeff((Double) value);
                    break;
                case "timeStep":
                    config.setTimeStep((Double) value);
                    break;
                case "maxSteps":
                    config.setMaxSteps((Integer) value);
                    break;
                default:
                    System.err.println("Unknown parameter: " + key);
            }
        }
    }

    /**
     * 获取所有注册的场景类型
     */
    public ScenarioType[] getRegisteredScenarios() {
        return scenarioRegistry.keySet().toArray(new ScenarioType[0]);
    }

    /**
     * 创建场景的简化描述
     */
    public String getScenarioDescription(ScenarioType type) {
        EnvironmentConfig config = loadScenario(type);
        return String.format("%s: %d车道, 限速%.0f km/h, 车辆密度%d辆/公里",
                type.getName(),
                config.getLaneCount(),
                config.getSpeedLimit(),
                config.getVehicleDensity());
    }
}
