package io.leavesfly.tinyai.agent.embodied.model;

/**
 * 传感器类型枚举
 * 定义具身智能体支持的传感器类型
 *
 * @author TinyAI Team
 */
public enum SensorType {
    /**
     * 相机传感器 - 提供RGB图像数据
     */
    CAMERA("相机传感器", "RGB图像"),

    /**
     * 激光雷达 - 提供3D点云数据
     */
    LIDAR("激光雷达", "3D点云"),

    /**
     * IMU惯性测量单元 - 提供加速度和角速度
     */
    IMU("惯性测量单元", "加速度+角速度"),

    /**
     * GPS定位 - 提供全局坐标
     */
    GPS("GPS定位", "经纬度坐标"),

    /**
     * 速度传感器 - 提供当前速度
     */
    SPEEDOMETER("速度传感器", "速度标量");

    private final String name;
    private final String outputType;

    SensorType(String name, String outputType) {
        this.name = name;
        this.outputType = outputType;
    }

    public String getName() {
        return name;
    }

    public String getOutputType() {
        return outputType;
    }
}
