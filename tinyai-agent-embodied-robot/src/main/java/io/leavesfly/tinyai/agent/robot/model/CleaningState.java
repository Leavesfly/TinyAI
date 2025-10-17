package io.leavesfly.tinyai.agent.robot.model;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import java.util.ArrayList;
import java.util.List;

/**
 * 清扫状态类
 * 
 * <p>表示扫地机器人在某一时刻的完整观测状态，包括感知信息和内部状态。</p>
 * 
 * @author TinyAI Team
 */
public class CleaningState {
    /**
     * 视觉特征向量（256维）
     */
    private NdArray visualFeatures;
    
    /**
     * 激光雷达特征（128维）
     */
    private NdArray lidarFeatures;
    
    /**
     * 机器人自身状态
     */
    private RobotState robotState;
    
    /**
     * 障碍物列表
     */
    private List<ObstacleInfo> obstacleMap;
    
    /**
     * 地面清扫地图
     */
    private FloorMap floorMap;
    
    /**
     * 充电站信息
     */
    private ChargingStationInfo chargingStationInfo;
    
    /**
     * 时间戳
     */
    private long timestamp;
    
    /**
     * 构造函数
     */
    public CleaningState() {
        this.visualFeatures = NdArray.zeros(Shape.of(256));
        this.lidarFeatures = NdArray.zeros(Shape.of(128));
        this.robotState = new RobotState();
        this.obstacleMap = new ArrayList<>();
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 获取完整的状态向量（用于神经网络输入）
     * 
     * @return 状态向量
     */
    public NdArray getStateVector() {
        // 组合所有特征：256(视觉) + 128(雷达) + 16(机器人状态) = 400维
        float[] stateData = new float[400];
        
        // 视觉特征
        for (int i = 0; i < 256; i++) {
            stateData[i] = visualFeatures.get(i);
        }
        
        // 雷达特征
        for (int i = 0; i < 128; i++) {
            stateData[256 + i] = lidarFeatures.get(i);
        }
        
        // 机器人状态（归一化）
        int offset = 384;
        stateData[offset++] = (float) (robotState.getPosition().getX() / 10.0); // 假设最大10米
        stateData[offset++] = (float) (robotState.getPosition().getY() / 10.0);
        stateData[offset++] = (float) (robotState.getHeading() / (2 * Math.PI));
        stateData[offset++] = (float) (robotState.getLinearSpeed() / 0.5);
        stateData[offset++] = (float) (robotState.getAngularSpeed() / (Math.PI / 2));
        stateData[offset++] = (float) (robotState.getBatteryLevel() / 100.0);
        stateData[offset++] = (float) (robotState.getDustCapacity() / 100.0);
        stateData[offset++] = (float) (robotState.getBrushSpeed() / 5000.0);
        stateData[offset++] = robotState.isCleaning() ? 1.0f : 0.0f;
        
        // 充电站信息
        if (chargingStationInfo != null) {
            stateData[offset++] = (float) (chargingStationInfo.getDistance() / 10.0);
            stateData[offset++] = (float) (chargingStationInfo.getDirection() / Math.PI);
        } else {
            stateData[offset++] = 1.0f; // 最大距离
            stateData[offset++] = 0.0f;
        }
        
        // 地图覆盖率
        if (floorMap != null) {
            stateData[offset++] = (float) floorMap.getCoverageRate();
            stateData[offset++] = (float) (floorMap.getTotalDust() / (floorMap.getWidth() * floorMap.getHeight()));
        } else {
            stateData[offset++] = 0.0f;
            stateData[offset++] = 1.0f;
        }
        
        // 最近障碍物距离
        double minObstacleDistance = 10.0;
        if (!obstacleMap.isEmpty()) {
            for (ObstacleInfo obstacle : obstacleMap) {
                minObstacleDistance = Math.min(minObstacleDistance, obstacle.getDistance());
            }
        }
        stateData[offset++] = (float) (minObstacleDistance / 10.0);
        
        // 障碍物数量（归一化）
        stateData[offset++] = (float) Math.min(1.0, obstacleMap.size() / 20.0);
        
        return NdArray.of(stateData, Shape.of(400));
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
    
    public RobotState getRobotState() {
        return robotState;
    }
    
    public void setRobotState(RobotState robotState) {
        this.robotState = robotState;
    }
    
    public List<ObstacleInfo> getObstacleMap() {
        return obstacleMap;
    }
    
    public void setObstacleMap(List<ObstacleInfo> obstacleMap) {
        this.obstacleMap = obstacleMap;
    }
    
    public FloorMap getFloorMap() {
        return floorMap;
    }
    
    public void setFloorMap(FloorMap floorMap) {
        this.floorMap = floorMap;
    }
    
    public ChargingStationInfo getChargingStationInfo() {
        return chargingStationInfo;
    }
    
    public void setChargingStationInfo(ChargingStationInfo chargingStationInfo) {
        this.chargingStationInfo = chargingStationInfo;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    @Override
    public String toString() {
        return String.format("CleaningState(robot=%s, obstacles=%d, coverage=%.1f%%)",
                             robotState, obstacleMap.size(), 
                             floorMap != null ? floorMap.getCoverageRate() * 100 : 0.0);
    }
}
