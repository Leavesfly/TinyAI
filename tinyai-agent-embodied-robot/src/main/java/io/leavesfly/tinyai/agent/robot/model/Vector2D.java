package io.leavesfly.tinyai.agent.robot.model;

/**
 * 二维向量类
 * 
 * <p>表示二维空间中的点或向量，提供基本的向量运算。</p>
 * 
 * @author TinyAI Team
 */
public class Vector2D {
    /**
     * X坐标（米）
     */
    private double x;
    
    /**
     * Y坐标（米）
     */
    private double y;
    
    /**
     * 构造函数
     * 
     * @param x X坐标
     * @param y Y坐标
     */
    public Vector2D(double x, double y) {
        this.x = x;
        this.y = y;
    }
    
    /**
     * 默认构造函数（原点）
     */
    public Vector2D() {
        this(0.0, 0.0);
    }
    
    /**
     * 拷贝构造函数
     * 
     * @param other 要拷贝的向量
     */
    public Vector2D(Vector2D other) {
        this(other.x, other.y);
    }
    
    /**
     * 计算向量长度（模）
     * 
     * @return 向量长度
     */
    public double magnitude() {
        return Math.sqrt(x * x + y * y);
    }
    
    /**
     * 计算与另一个点的距离
     * 
     * @param other 另一个点
     * @return 距离
     */
    public double distanceTo(Vector2D other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * 向量加法
     * 
     * @param other 另一个向量
     * @return 新向量
     */
    public Vector2D add(Vector2D other) {
        return new Vector2D(this.x + other.x, this.y + other.y);
    }
    
    /**
     * 向量减法
     * 
     * @param other 另一个向量
     * @return 新向量
     */
    public Vector2D subtract(Vector2D other) {
        return new Vector2D(this.x - other.x, this.y - other.y);
    }
    
    /**
     * 标量乘法
     * 
     * @param scalar 标量
     * @return 新向量
     */
    public Vector2D multiply(double scalar) {
        return new Vector2D(this.x * scalar, this.y * scalar);
    }
    
    /**
     * 归一化（单位向量）
     * 
     * @return 归一化后的向量
     */
    public Vector2D normalize() {
        double mag = magnitude();
        if (mag < 1e-10) {
            return new Vector2D(0, 0);
        }
        return new Vector2D(x / mag, y / mag);
    }
    
    /**
     * 计算与X轴的夹角
     * 
     * @return 角度（弧度）
     */
    public double angle() {
        return Math.atan2(y, x);
    }
    
    /**
     * 点积运算
     * 
     * @param other 另一个向量
     * @return 点积结果
     */
    public double dot(Vector2D other) {
        return this.x * other.x + this.y * other.y;
    }
    
    // Getters and Setters
    public double getX() {
        return x;
    }
    
    public void setX(double x) {
        this.x = x;
    }
    
    public double getY() {
        return y;
    }
    
    public void setY(double y) {
        this.y = y;
    }
    
    @Override
    public String toString() {
        return String.format("Vector2D(x=%.3f, y=%.3f)", x, y);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Vector2D other = (Vector2D) obj;
        return Math.abs(x - other.x) < 1e-6 && Math.abs(y - other.y) < 1e-6;
    }
    
    @Override
    public int hashCode() {
        long xBits = Double.doubleToLongBits(x);
        long yBits = Double.doubleToLongBits(y);
        return (int) (xBits ^ yBits);
    }
}
