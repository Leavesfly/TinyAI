package io.leavesfly.tinyai.agent.robot.model;

/**
 * 障碍物信息类
 * 
 * <p>表示环境中的障碍物信息，包括类型、位置、大小等。</p>
 * 
 * @author TinyAI Team
 */
public class ObstacleInfo {
    /**
     * 障碍物类型
     */
    private ObstacleType type;
    
    /**
     * 障碍物位置
     */
    private Vector2D position;
    
    /**
     * 包围盒
     */
    private BoundingBox boundingBox;
    
    /**
     * 与机器人的距离
     */
    private double distance;
    
    /**
     * 相对角度（弧度）
     */
    private double relativeAngle;
    
    /**
     * 是否移动（如宠物）
     */
    private boolean isMoving;
    
    /**
     * 构造函数
     * 
     * @param type 障碍物类型
     * @param position 位置
     * @param boundingBox 包围盒
     */
    public ObstacleInfo(ObstacleType type, Vector2D position, BoundingBox boundingBox) {
        this.type = type;
        this.position = position;
        this.boundingBox = boundingBox;
        this.distance = 0.0;
        this.relativeAngle = 0.0;
        this.isMoving = (type == ObstacleType.PET);
    }
    
    /**
     * 更新相对于机器人的信息
     * 
     * @param robotPosition 机器人位置
     * @param robotHeading 机器人朝向
     */
    public void updateRelativeInfo(Vector2D robotPosition, double robotHeading) {
        // 计算距离
        this.distance = position.distanceTo(robotPosition);
        
        // 计算相对角度
        Vector2D diff = position.subtract(robotPosition);
        double absoluteAngle = diff.angle();
        this.relativeAngle = absoluteAngle - robotHeading;
        
        // 归一化到 [-π, π]
        while (this.relativeAngle > Math.PI) {
            this.relativeAngle -= 2 * Math.PI;
        }
        while (this.relativeAngle < -Math.PI) {
            this.relativeAngle += 2 * Math.PI;
        }
    }
    
    // Getters and Setters
    public ObstacleType getType() {
        return type;
    }
    
    public void setType(ObstacleType type) {
        this.type = type;
    }
    
    public Vector2D getPosition() {
        return position;
    }
    
    public void setPosition(Vector2D position) {
        this.position = position;
    }
    
    public BoundingBox getBoundingBox() {
        return boundingBox;
    }
    
    public void setBoundingBox(BoundingBox boundingBox) {
        this.boundingBox = boundingBox;
    }
    
    public double getDistance() {
        return distance;
    }
    
    public void setDistance(double distance) {
        this.distance = distance;
    }
    
    public double getRelativeAngle() {
        return relativeAngle;
    }
    
    public void setRelativeAngle(double relativeAngle) {
        this.relativeAngle = relativeAngle;
    }
    
    public boolean isMoving() {
        return isMoving;
    }
    
    public void setMoving(boolean moving) {
        isMoving = moving;
    }
    
    @Override
    public String toString() {
        return String.format("ObstacleInfo(type=%s, pos=%s, distance=%.2f, angle=%.2f)",
                             type, position, distance, relativeAngle);
    }
}
