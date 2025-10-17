package io.leavesfly.tinyai.agent.embodied.perception;

import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;
import io.leavesfly.tinyai.agent.embodied.model.ObstacleInfo;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 特征提取器
 * 从原始感知数据中提取特征
 *
 * @author TinyAI Team
 */
public class FeatureExtractor {
    private static final int VISUAL_FEATURE_DIM = 256;
    private static final int LIDAR_FEATURE_DIM = 128;

    /**
     * 提取视觉特征
     */
    public NdArray extractVisualFeatures(PerceptionState state) {
        float[] features = new float[VISUAL_FEATURE_DIM];

        // 基础特征：车道信息
        if (state.getLaneInfo() != null) {
            features[0] = (float) state.getLaneInfo().getLateralDeviation();
            features[1] = (float) state.getLaneInfo().getCurvature();
            features[2] = state.getLaneInfo().isLeftLaneAvailable() ? 1.0f : 0.0f;
            features[3] = state.getLaneInfo().isRightLaneAvailable() ? 1.0f : 0.0f;
        }

        // 车辆状态特征
        if (state.getVehicleState() != null) {
            features[4] = (float) state.getVehicleState().getSpeed();
            features[5] = (float) state.getVehicleState().getAcceleration();
            features[6] = (float) state.getVehicleState().getHeading();
            features[7] = (float) state.getVehicleState().getSteeringAngle();
        }

        // 障碍物特征（编码最近的几个障碍物）
        if (state.getObstacleMap() != null && !state.getObstacleMap().isEmpty()) {
            int idx = 8;
            for (int i = 0; i < Math.min(5, state.getObstacleMap().size()) && idx < VISUAL_FEATURE_DIM - 4; i++) {
                ObstacleInfo obs = state.getObstacleMap().get(i);
                features[idx++] = (float) obs.getDistance();
                features[idx++] = (float) obs.getPosition().getX();
                features[idx++] = (float) obs.getPosition().getY();
                features[idx++] = (float) obs.getVelocity().magnitude();
            }
        }

        return NdArray.of(features);
    }

    /**
     * 提取激光雷达特征
     */
    public NdArray extractLidarFeatures(PerceptionState state) {
        float[] features = new float[LIDAR_FEATURE_DIM];

        // 简化的距离特征
        if (state.getObstacleMap() != null && !state.getObstacleMap().isEmpty()) {
            ObstacleInfo nearest = state.getNearestObstacle();
            if (nearest != null) {
                features[0] = (float) nearest.getDistance();
                features[1] = (float) nearest.getPosition().getX();
                features[2] = (float) nearest.getPosition().getY();
                features[3] = (float) nearest.getConfidence();
            }

            // 统计特征
            features[4] = state.getObstacleMap().size();
            features[5] = (float) state.countObstaclesInRange(50.0);
            features[6] = (float) state.countObstaclesInRange(20.0);
        }

        return NdArray.of(features);
    }

    /**
     * 多模态特征融合
     */
    public NdArray fuseFeatures(NdArray visual, NdArray lidar) {
        // 简单拼接融合
        int totalDim = VISUAL_FEATURE_DIM + LIDAR_FEATURE_DIM;
        float[] fused = new float[totalDim];

        // 拷贝视觉特征
        for (int i = 0; i < VISUAL_FEATURE_DIM; i++) {
            fused[i] = visual.get(i);
        }

        // 拷贝激光雷达特征
        for (int i = 0; i < LIDAR_FEATURE_DIM; i++) {
            fused[VISUAL_FEATURE_DIM + i] = lidar.get(i);
        }

        return NdArray.of(fused);
    }
}
