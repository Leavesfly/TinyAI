package io.leavesfly.tinyai.agent.embodied;

import io.leavesfly.tinyai.agent.embodied.env.DrivingEnvironment;
import io.leavesfly.tinyai.agent.embodied.env.EnvironmentConfig;
import io.leavesfly.tinyai.agent.embodied.env.impl.SimpleDrivingEnv;
import io.leavesfly.tinyai.agent.embodied.model.*;

/**
 * 具身智能体简单演示
 * 演示环境的基本使用
 *
 * @author TinyAI Team
 */
public class SimpleDemo {
    public static void main(String[] args) {
        System.out.println("=== TinyAI 具身智能体演示 ===\n");
        
        // 1. 创建测试环境
        EnvironmentConfig config = EnvironmentConfig.createTestConfig();
        DrivingEnvironment env = new SimpleDrivingEnv(config);
        
        System.out.println("环境配置: " + config);
        System.out.println();
        
        // 2. 重置环境
        PerceptionState state = env.reset();
        System.out.println("初始状态: " + state);
        System.out.println();
        
        // 3. 运行几步简单的固定策略
        System.out.println("=== 开始模拟 ===");
        
        for (int step = 0; step < 50; step++) {
            // 简单的直行策略：保持油门，无转向
            DrivingAction action = new DrivingAction(0.0, 0.3, 0.0);
            
            // 执行动作
            StepResult result = env.step(action);
            
            // 每10步打印一次
            if (step % 10 == 0) {
                System.out.println(env.render());
                System.out.println("奖励: " + String.format("%.3f", result.getReward()));
                System.out.println();
            }
            
            // 如果终止，退出
            if (result.isDone()) {
                System.out.println("情景终止于步骤 " + step);
                System.out.println("最终状态: " + result.getObservation());
                break;
            }
        }
        
        // 4. 清理
        env.close();
        
        System.out.println("\n=== 演示结束 ===");
    }
}
