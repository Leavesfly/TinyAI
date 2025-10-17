package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 车道几何信息类
 * 描述车道的几何结构和车辆在车道中的位置
 *
 * @author TinyAI Team
 */
public class LaneGeometry {
    private int laneId;                // 车道ID
    private double laneWidth;          // 车道宽度
    private double lateralDeviation;   // 横向偏离距离
    private double curvature;          // 曲率
    private boolean isLeftLaneAvailable;   // 左侧车道是否可用
    private boolean isRightLaneAvailable;  // 右侧车道是否可用

    public LaneGeometry(int laneId, double laneWidth) {
        this.laneId = laneId;
        this.laneWidth = laneWidth;
        this.lateralDeviation = 0.0;
        this.curvature = 0.0;
        this.isLeftLaneAvailable = false;
        this.isRightLaneAvailable = false;
    }

    /**
     * 判断车辆是否在车道中心
     */
    public boolean isInLaneCenter(double threshold) {
        return Math.abs(lateralDeviation) < threshold;
    }

    /**
     * 获取偏离车道中心的程度（归一化到[-1, 1]）
     */
    public double getNormalizedDeviation() {
        return Math.max(-1.0, Math.min(1.0, lateralDeviation / (laneWidth / 2.0)));
    }

    // Getters and Setters
    public int getLaneId() {
        return laneId;
    }

    public void setLaneId(int laneId) {
        this.laneId = laneId;
    }

    public double getLaneWidth() {
        return laneWidth;
    }

    public void setLaneWidth(double laneWidth) {
        this.laneWidth = laneWidth;
    }

    public double getLateralDeviation() {
        return lateralDeviation;
    }

    public void setLateralDeviation(double lateralDeviation) {
        this.lateralDeviation = lateralDeviation;
    }

    public double getCurvature() {
        return curvature;
    }

    public void setCurvature(double curvature) {
        this.curvature = curvature;
    }

    public boolean isLeftLaneAvailable() {
        return isLeftLaneAvailable;
    }

    public void setLeftLaneAvailable(boolean leftLaneAvailable) {
        isLeftLaneAvailable = leftLaneAvailable;
    }

    public boolean isRightLaneAvailable() {
        return isRightLaneAvailable;
    }

    public void setRightLaneAvailable(boolean rightLaneAvailable) {
        isRightLaneAvailable = rightLaneAvailable;
    }

    @Override
    public String toString() {
        return String.format("Lane[id=%d, width=%.2f, deviation=%.2f]",
                laneId, laneWidth, lateralDeviation);
    }
}
