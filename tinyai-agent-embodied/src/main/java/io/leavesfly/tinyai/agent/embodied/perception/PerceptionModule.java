package io.leavesfly.tinyai.agent.embodied.perception;

import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.sensor.SensorSuite;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 感知模块
 * 负责处理传感器数据并生成统一的感知状态
 *
 * @author TinyAI Team
 */
public class PerceptionModule {
    private SensorSuite sensorSuite;
    private FeatureExtractor featureExtractor;
    private boolean initialized;

    public PerceptionModule(SensorSuite sensorSuite) {
        this.sensorSuite = sensorSuite;
        this.featureExtractor = new FeatureExtractor();
        this.initialized = false;
    }

    /**
     * 初始化感知模块
     */
    public void initialize() {
        if (!sensorSuite.isInitialized()) {
            sensorSuite.initialize();
        }
        initialized = true;
    }

    /**
     * 处理感知并生成状态
     */
    public PerceptionState process(PerceptionState rawState) {
        if (!initialized) {
            initialize();
        }

        // 提取特征
        NdArray visualFeatures = featureExtractor.extractVisualFeatures(rawState);
        NdArray lidarFeatures = featureExtractor.extractLidarFeatures(rawState);

        // 更新感知状态
        rawState.setVisualFeatures(visualFeatures);
        rawState.setLidarFeatures(lidarFeatures);

        return rawState;
    }

    /**
     * 重置感知模块
     */
    public void reset() {
        sensorSuite.resetAll();
    }

    public boolean isInitialized() {
        return initialized;
    }

    public SensorSuite getSensorSuite() {
        return sensorSuite;
    }
}
