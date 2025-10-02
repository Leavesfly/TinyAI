package io.leavesfly.tinyai.util;

/**
 * 简单的配置类
 * 
 * @author leavesfly
 * @version 0.01
 * 
 * Config类用于存储全局配置信息，包括训练模式和激活函数类型枚举。
 */
public class Config {

    /**
     * 训练模式开关
     * true表示处于训练模式，false表示处于推理模式
     */
    public static Boolean train = true;

    /**
     * 激活函数类型枚举
     * 定义了框架支持的激活函数类型
     */
    public enum ActiveFunc {
        /**
         * ReLU激活函数
         */
        ReLU, 
        
        /**
         * Sigmoid激活函数
         */
        Sigmoid, 
        
        /**
         * SoftMax激活函数
         */
        SoftMax, 
        
        /**
         * Tanh激活函数
         */
        Tanh
    }
}