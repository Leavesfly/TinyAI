package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 障碍物信息类
 * 包含障碍物的类型、位置、速度等完整信息
 *
 * @author TinyAI Team
 */
public class ObstacleInfo {
    private ObstacleType objectType;
    private Vector3D position;        // 相对位置
    private Vector3D velocity;        // 相对速度
    private BoundingBox boundingBox;  // 包围盒
    private double confidence;        // 检测置信度

    public ObstacleInfo(ObstacleType objectType, Vector3D position, Vector3D velocity,
                       BoundingBox boundingBox, double confidence) {
        this.objectType = objectType;
        this.position = position;
        this.velocity = velocity;
        this.boundingBox = boundingBox;
        this.confidence = confidence;
    }

    /**
     * 计算障碍物距离
     */
    public double getDistance() {
        return position.magnitude();
    }

    /**
     * 判断是否为危险障碍物（距离近且速度快）
     */
    public boolean isDangerous(double distanceThreshold, double velocityThreshold) {
        return getDistance() < distanceThreshold &&
               velocity.magnitude() > velocityThreshold;
    }

    // Getters and Setters
    public ObstacleType getObjectType() {
        return objectType;
    }

    public void setObjectType(ObstacleType objectType) {
        this.objectType = objectType;
    }

    public Vector3D getPosition() {
        return position;
    }

    public void setPosition(Vector3D position) {
        this.position = position;
    }

    public Vector3D getVelocity() {
        return velocity;
    }

    public void setVelocity(Vector3D velocity) {
        this.velocity = velocity;
    }

    public BoundingBox getBoundingBox() {
        return boundingBox;
    }

    public void setBoundingBox(BoundingBox boundingBox) {
        this.boundingBox = boundingBox;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    @Override
    public String toString() {
        return String.format("Obstacle[type=%s, pos=%s, dist=%.2f, conf=%.2f]",
                objectType.getName(), position, getDistance(), confidence);
    }
}
