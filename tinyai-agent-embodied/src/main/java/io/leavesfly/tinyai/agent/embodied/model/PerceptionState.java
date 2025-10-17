package io.leavesfly.tinyai.agent.embodied.model;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;

/**
 * 感知状态类
 * 整合所有传感器信息，提供完整的环境感知
 *
 * @author TinyAI Team
 */
public class PerceptionState {
    private NdArray visualFeatures;      // 视觉特征向量
    private NdArray lidarFeatures;       // 激光雷达特征
    private VehicleState vehicleState;   // 车辆自身状态
    private List<ObstacleInfo> obstacleMap;  // 障碍物列表
    private LaneGeometry laneInfo;       // 车道信息
    private long timestamp;              // 时间戳

    public PerceptionState() {
        this.obstacleMap = new ArrayList<>();
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * 获取最近的障碍物
     */
    public ObstacleInfo getNearestObstacle() {
        if (obstacleMap == null || obstacleMap.isEmpty()) {
            return null;
        }

        ObstacleInfo nearest = obstacleMap.get(0);
        double minDistance = nearest.getDistance();

        for (ObstacleInfo obstacle : obstacleMap) {
            double distance = obstacle.getDistance();
            if (distance < minDistance) {
                minDistance = distance;
                nearest = obstacle;
            }
        }

        return nearest;
    }

    /**
     * 统计指定范围内的障碍物数量
     */
    public int countObstaclesInRange(double range) {
        int count = 0;
        for (ObstacleInfo obstacle : obstacleMap) {
            if (obstacle.getDistance() <= range) {
                count++;
            }
        }
        return count;
    }

    /**
     * 获取危险障碍物列表
     */
    public List<ObstacleInfo> getDangerousObstacles(double distanceThreshold, double velocityThreshold) {
        List<ObstacleInfo> dangerous = new ArrayList<>();
        for (ObstacleInfo obstacle : obstacleMap) {
            if (obstacle.isDangerous(distanceThreshold, velocityThreshold)) {
                dangerous.add(obstacle);
            }
        }
        return dangerous;
    }

    // Getters and Setters
    public NdArray getVisualFeatures() {
        return visualFeatures;
    }

    public void setVisualFeatures(NdArray visualFeatures) {
        this.visualFeatures = visualFeatures;
    }

    public NdArray getLidarFeatures() {
        return lidarFeatures;
    }

    public void setLidarFeatures(NdArray lidarFeatures) {
        this.lidarFeatures = lidarFeatures;
    }

    public VehicleState getVehicleState() {
        return vehicleState;
    }

    public void setVehicleState(VehicleState vehicleState) {
        this.vehicleState = vehicleState;
    }

    public List<ObstacleInfo> getObstacleMap() {
        return obstacleMap;
    }

    public void setObstacleMap(List<ObstacleInfo> obstacleMap) {
        this.obstacleMap = obstacleMap;
    }

    public LaneGeometry getLaneInfo() {
        return laneInfo;
    }

    public void setLaneInfo(LaneGeometry laneInfo) {
        this.laneInfo = laneInfo;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    @Override
    public String toString() {
        return String.format("PerceptionState[vehicle=%s, obstacles=%d, lane=%s]",
                vehicleState, obstacleMap.size(), laneInfo);
    }
}
