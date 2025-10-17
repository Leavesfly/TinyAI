package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 包围盒类
 * 用于表示障碍物的空间占用
 *
 * @author TinyAI Team
 */
public class BoundingBox {
    private double length;  // 长度
    private double width;   // 宽度
    private double height;  // 高度

    public BoundingBox(double length, double width, double height) {
        this.length = length;
        this.width = width;
        this.height = height;
    }

    /**
     * 计算包围盒体积
     */
    public double getVolume() {
        return length * width * height;
    }

    /**
     * 计算包围盒占地面积
     */
    public double getFootprint() {
        return length * width;
    }

    // Getters and Setters
    public double getLength() {
        return length;
    }

    public void setLength(double length) {
        this.length = length;
    }

    public double getWidth() {
        return width;
    }

    public void setWidth(double width) {
        this.width = width;
    }

    public double getHeight() {
        return height;
    }

    public void setHeight(double height) {
        this.height = height;
    }

    @Override
    public String toString() {
        return String.format("BoundingBox(L=%.2f, W=%.2f, H=%.2f)", length, width, height);
    }
}
