package io.leavesfly.tinyai.agent.robot;

import io.leavesfly.tinyai.agent.robot.env.*;
import io.leavesfly.tinyai.agent.robot.model.*;

/**
 * 简单演示程序
 * 
 * <p>展示扫地机器人的基本功能。</p>
 * 
 * @author TinyAI Team
 */
public class SimpleDemo {
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("TinyAI 具身智能扫地机器人 - 简单演示");
        System.out.println("=".repeat(60));
        System.out.println();
        
        // 创建环境
        System.out.println("1. 创建清扫环境...");
        EnvironmentConfig config = EnvironmentConfig.createSimpleRoomConfig();
        System.out.println("   " + config);
        
        CleaningEnvironment env = new SimpleCleaningEnv(config);
        System.out.println("   环境创建成功！");
        System.out.println();
        
        // 重置环境
        System.out.println("2. 初始化环境...");
        CleaningState initialState = env.reset();
        RobotState robotState = initialState.getRobotState();
        System.out.println("   机器人初始状态: " + robotState);
        System.out.println("   初始覆盖率: " + String.format("%.1f%%", 
                           initialState.getFloorMap().getCoverageRate() * 100));
        System.out.println();
        
        // 执行预定义动作序列
        System.out.println("3. 执行清扫任务...");
        System.out.println("   " + "-".repeat(50));
        
        int maxSteps = 50;
        double totalReward = 0.0;
        
        for (int step = 0; step < maxSteps; step++) {
            // 简单策略：前进 + 偶尔转向
            CleaningAction action;
            if (step % 10 == 9) {
                // 每10步转向一次
                action = CleaningAction.turnRight(0.5);
            } else {
                // 其他时间前进并清扫
                action = CleaningAction.moveForward(0.5);
            }
            
            // 执行动作
            StepResult result = env.step(action);
            totalReward += result.getReward();
            
            // 打印进度
            if (step % 10 == 0) {
                CleaningState state = result.getObservation();
                RobotState robot = state.getRobotState();
                double coverage = state.getFloorMap().getCoverageRate();
                
                System.out.println(String.format(
                    "   步骤 %3d | 位置: (%.2f, %.2f) | 电量: %5.1f%% | 覆盖率: %5.1f%% | 奖励: %7.2f",
                    step, 
                    robot.getPosition().getX(),
                    robot.getPosition().getY(),
                    robot.getBatteryLevel(),
                    coverage * 100,
                    result.getReward()
                ));
            }
            
            // 检查是否结束
            if (result.isDone()) {
                System.out.println();
                System.out.println("   任务结束！");
                System.out.println("   原因: " + getTerminationReason(result));
                break;
            }
        }
        
        System.out.println("   " + "-".repeat(50));
        System.out.println();
        
        // 打印最终统计
        System.out.println("4. 最终统计结果:");
        CleaningState finalState = env.getObservation();
        RobotState finalRobot = finalState.getRobotState();
        FloorMap finalMap = finalState.getFloorMap();
        
        System.out.println("   总奖励: " + String.format("%.2f", totalReward));
        System.out.println("   最终覆盖率: " + String.format("%.1f%%", 
                           finalMap.getCoverageRate() * 100));
        System.out.println("   剩余电量: " + String.format("%.1f%%", 
                           finalRobot.getBatteryLevel()));
        System.out.println("   尘盒容量: " + String.format("%.1f%%", 
                           finalRobot.getDustCapacity()));
        System.out.println();
        
        // 关闭环境
        env.close();
        
        System.out.println("=".repeat(60));
        System.out.println("演示完成！");
        System.out.println("=".repeat(60));
    }
    
    /**
     * 获取终止原因
     */
    private static String getTerminationReason(StepResult result) {
        CleaningState state = result.getObservation();
        RobotState robot = state.getRobotState();
        
        if (state.getFloorMap().getCoverageRate() >= 0.95) {
            return "达到目标覆盖率";
        } else if (robot.getBatteryLevel() <= 0) {
            return "电量耗尽";
        } else {
            return "超过最大步数";
        }
    }
}
