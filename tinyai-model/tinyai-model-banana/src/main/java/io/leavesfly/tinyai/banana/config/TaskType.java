package io.leavesfly.tinyai.banana.config;

import java.util.Optional;

/**

/**
 * Banana模型支持的任务类型
 * 
 * 定义模型可以处理的不同类型任务,用于任务感知路由和优化。
 * 
 * @author leavesfly
 * @version 1.0
 */
public enum TaskType {
    
    /**
     * 文本到图像生成任务
     * 根据文本描述生成对应图像
     */
    TEXT_TO_IMAGE("text2image", "文本到图像生成"),
    
    /**
     * 图像编辑任务
     * 根据文本指令编辑现有图像
     */
    IMAGE_EDITING("image_edit", "图像编辑"),
    
    /**
     * 图像理解任务
     * 理解和描述图像内容
     */
    IMAGE_UNDERSTANDING("image_understand", "图像理解"),
    
    /**
     * 多图像组合任务
     * 将多张图像组合成新图像
     */
    MULTI_IMAGE_COMPOSITION("multi_compose", "多图像组合"),
    
    /**
     * 通用多模态任务
     * 文本和图像的通用交互
     */
    GENERAL_MULTIMODAL("general", "通用多模态");
    
    private final String code;
    private final String description;
    
    TaskType(String code, String description) {
        this.code = code;
        this.description = description;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getDescription() {
        return description;
    }
    
    /**
     * 根据code获取TaskType
     * 
     * @param code 任务类型代码
     * @return 匹配的TaskType，未找到时返回空Optional
     */
    public static Optional<TaskType> fromCode(String code) {
        if (code == null) {
            return Optional.empty();
        }
        for (TaskType type : values()) {
            if (type.code.equals(code)) {
                return Optional.of(type);
            }
        }
        return Optional.empty();
    }
    
    @Override
    public String toString() {
        return description + " (" + code + ")";
    }
}
