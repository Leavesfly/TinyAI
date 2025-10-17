package io.leavesfly.tinyai.agent.robot.model;

/**
 * 包围盒类
 * 
 * <p>表示三维空间中的轴对齐包围盒，用于障碍物和机器人的碰撞检测。</p>
 * 
 * @author TinyAI Team
 */
public class BoundingBox {
    /**
     * 宽度（米）
     */
    private double width;
    
    /**
     * 长度（米）
     */
    private double length;
    
    /**
     * 高度（米）
     */
    private double height;
    
    /**
     * 构造函数
     * 
     * @param width 宽度
     * @param length 长度
     * @param height 高度
     */
    public BoundingBox(double width, double length, double height) {
        this.width = width;
        this.length = length;
        this.height = height;
    }
    
    /**
     * 默认构造函数
     */
    public BoundingBox() {
        this(0.0, 0.0, 0.0);
    }
    
    /**
     * 计算占地面积
     * 
     * @return 面积（平方米）
     */
    public double getArea() {
        return width * length;
    }
    
    /**
     * 计算体积
     * 
     * @return 体积（立方米）
     */
    public double getVolume() {
        return width * length * height;
    }
    
    /**
     * 判断点是否在包围盒内部（二维投影）
     * 
     * @param point 待检测的点
     * @param center 包围盒中心
     * @return 是否在内部
     */
    public boolean contains(Vector2D point, Vector2D center) {
        double halfWidth = width / 2.0;
        double halfLength = length / 2.0;
        
        return Math.abs(point.getX() - center.getX()) <= halfWidth &&
               Math.abs(point.getY() - center.getY()) <= halfLength;
    }
    
    // Getters and Setters
    public double getWidth() {
        return width;
    }
    
    public void setWidth(double width) {
        this.width = width;
    }
    
    public double getLength() {
        return length;
    }
    
    public void setLength(double length) {
        this.length = length;
    }
    
    public double getHeight() {
        return height;
    }
    
    public void setHeight(double height) {
        this.height = height;
    }
    
    @Override
    public String toString() {
        return String.format("BoundingBox(width=%.3f, length=%.3f, height=%.3f)", 
                             width, length, height);
    }
}
