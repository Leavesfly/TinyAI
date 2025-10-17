package io.leavesfly.tinyai.agent.vla.model;

/**
 * 动作类型枚举
 * 定义VLA智能体支持的基础动作类型
 * 
 * @author TinyAI
 */
public enum ActionType {
    /** 移动末端执行器 */
    MOVE_END_EFFECTOR("移动末端执行器"),
    
    /** 旋转关节 */
    ROTATE_JOINTS("旋转关节"),
    
    /** 抓取物体 */
    GRASP_OBJECT("抓取物体"),
    
    /** 释放物体 */
    RELEASE_OBJECT("释放物体"),
    
    /** 导航到目标点 */
    NAVIGATE_TO_TARGET("导航到目标点"),
    
    /** 等待 */
    WAIT("等待"),
    
    /** 语言输出 */
    SPEAK("语言输出");
    
    private final String description;
    
    ActionType(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
}
