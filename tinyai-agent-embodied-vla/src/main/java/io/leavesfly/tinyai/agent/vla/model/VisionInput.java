package io.leavesfly.tinyai.agent.vla.model;

import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 视觉输入数据模型
 * 封装VLA智能体的视觉模态输入
 * 
 * @author TinyAI
 */
public class VisionInput {
    
    /** RGB图像，维度 [H, W, 3] */
    private NdArray rgbImage;
    
    /** 深度图（可选），维度 [H, W, 1] */
    private NdArray depthImage;
    
    /** 视觉编码特征，维度 [196, 768] */
    private NdArray imageFeatures;
    
    /** 目标分割掩码，维度 [N, H, W] */
    private NdArray objectMasks;
    
    /** 采集时间戳 */
    private long timestamp;
    
    /**
     * 构造函数 - 仅RGB图像
     */
    public VisionInput(NdArray rgbImage) {
        this.rgbImage = rgbImage;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 构造函数 - RGB + 深度图
     */
    public VisionInput(NdArray rgbImage, NdArray depthImage) {
        this.rgbImage = rgbImage;
        this.depthImage = depthImage;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 完整构造函数
     */
    public VisionInput(NdArray rgbImage, NdArray depthImage, NdArray objectMasks) {
        this.rgbImage = rgbImage;
        this.depthImage = depthImage;
        this.objectMasks = objectMasks;
        this.timestamp = System.currentTimeMillis();
    }
    
    // Getters and Setters
    public NdArray getRgbImage() {
        return rgbImage;
    }
    
    public void setRgbImage(NdArray rgbImage) {
        this.rgbImage = rgbImage;
    }
    
    public NdArray getDepthImage() {
        return depthImage;
    }
    
    public void setDepthImage(NdArray depthImage) {
        this.depthImage = depthImage;
    }
    
    public NdArray getImageFeatures() {
        return imageFeatures;
    }
    
    public void setImageFeatures(NdArray imageFeatures) {
        this.imageFeatures = imageFeatures;
    }
    
    public NdArray getObjectMasks() {
        return objectMasks;
    }
    
    public void setObjectMasks(NdArray objectMasks) {
        this.objectMasks = objectMasks;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    @Override
    public String toString() {
        return "VisionInput{" +
                "rgbImageShape=" + (rgbImage != null ? rgbImage.getShape() : "null") +
                ", hasDepth=" + (depthImage != null) +
                ", hasFeatures=" + (imageFeatures != null) +
                ", hasMasks=" + (objectMasks != null) +
                ", timestamp=" + timestamp +
                '}';
    }
}
