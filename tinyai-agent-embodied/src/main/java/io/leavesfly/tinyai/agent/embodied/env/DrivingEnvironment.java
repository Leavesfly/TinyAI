package io.leavesfly.tinyai.agent.embodied.env;

import io.leavesfly.tinyai.agent.embodied.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * 驾驶环境接口
 * 定义驾驶模拟环境的标准操作
 *
 * @author TinyAI Team
 */
public interface DrivingEnvironment {
    /**
     * 重置环境到初始状态
     * @return 初始观测状态
     */
    PerceptionState reset();

    /**
     * 执行动作，环境状态向前推进一步
     * @param action 驾驶动作
     * @return 步进结果（包含下一状态、奖励、是否终止等）
     */
    StepResult step(DrivingAction action);

    /**
     * 获取当前观测
     * @return 当前感知状态
     */
    PerceptionState getObservation();

    /**
     * 判断是否终止
     * @return 是否到达终止状态
     */
    boolean isTerminated();

    /**
     * 获取指定传感器的数据
     * @param sensorType 传感器类型
     * @return 传感器数据
     */
    NdArray getSensorData(SensorType sensorType);

    /**
     * 渲染当前环境状态（可选，用于可视化）
     * @return 渲染数据
     */
    Object render();

    /**
     * 获取环境配置信息
     */
    EnvironmentConfig getConfig();

    /**
     * 设置环境配置
     */
    void setConfig(EnvironmentConfig config);

    /**
     * 获取当前场景类型
     */
    ScenarioType getScenarioType();

    /**
     * 关闭环境，释放资源
     */
    void close();
}
