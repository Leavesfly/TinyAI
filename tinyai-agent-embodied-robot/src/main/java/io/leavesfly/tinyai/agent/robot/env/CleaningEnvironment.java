package io.leavesfly.tinyai.agent.robot.env;

import io.leavesfly.tinyai.agent.robot.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 清扫环境接口
 * 
 * <p>定义清扫环境的标准行为，支持多种环境实现。</p>
 * 
 * @author TinyAI Team
 */
public interface CleaningEnvironment {
    /**
     * 重置环境到初始状态
     * 
     * @return 初始观测状态
     */
    CleaningState reset();
    
    /**
     * 执行一步动作
     * 
     * @param action 清扫动作
     * @return 步进结果（新状态、奖励、是否结束）
     */
    StepResult step(CleaningAction action);
    
    /**
     * 获取当前观测
     * 
     * @return 当前状态
     */
    CleaningState getObservation();
    
    /**
     * 判断是否终止
     * 
     * @return 是否结束
     */
    boolean isTerminated();
    
    /**
     * 获取传感器数据
     * 
     * @param type 传感器类型
     * @return 传感器数据
     */
    NdArray getSensorData(SensorType type);
    
    /**
     * 渲染环境（可选，用于可视化）
     * 
     * @return 渲染对象
     */
    Object render();
    
    /**
     * 关闭环境
     */
    void close();
    
    /**
     * 获取场景类型
     * 
     * @return 场景类型
     */
    ScenarioType getScenarioType();
    
    /**
     * 获取环境配置
     * 
     * @return 环境配置
     */
    EnvironmentConfig getConfig();
}
