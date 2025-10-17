package io.leavesfly.tinyai.agent.vla;

import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.agent.vla.env.TaskScenario;
import io.leavesfly.tinyai.agent.vla.learning.BehaviorCloningLearner;
import io.leavesfly.tinyai.agent.vla.learning.VLALearningEngine;
import io.leavesfly.tinyai.agent.vla.model.TaskConfig;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.agent.vla.model.VLAState;

/**
 * VLA智能体演示程序
 * 展示视觉-语言-动作具身智能系统的完整功能
 * 
 * @author TinyAI
 */
public class VLADemo {
    
    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("  TinyAI VLA Agent Demonstration");
        System.out.println("  Vision-Language-Action Intelligence");
        System.out.println("========================================\n");
        
        // 1. 创建VLA智能体
        System.out.println("1. Creating VLA Agent...");
        VLAAgent agent = new VLAAgent(
            768,   // hiddenDim
            8,     // numHeads
            6,     // numLayers
            7      // actionDim (末端执行器7自由度)
        );
        agent.printModelInfo();
        
        // 2. 创建任务环境
        System.out.println("2. Creating Robot Environment...");
        TaskConfig taskConfig = new TaskConfig();
        taskConfig.setTaskName(TaskScenario.PICK_AND_PLACE.getName());
        taskConfig.setTaskDescription(TaskScenario.PICK_AND_PLACE.getDescription());
        taskConfig.setMaxSteps(100);
        taskConfig.setSuccessReward(100.0);
        taskConfig.setRender(false);
        
        RobotEnvironment env = new SimpleRobotEnv(taskConfig);
        System.out.println("Task: " + TaskScenario.PICK_AND_PLACE.getName());
        System.out.println("Difficulty: " + TaskScenario.PICK_AND_PLACE.getDifficultyStars());
        System.out.println("Description: " + TaskScenario.PICK_AND_PLACE.getDescription());
        System.out.println();
        
        // 3. 演示单个Episode
        System.out.println("3. Running Demo Episode...");
        runDemoEpisode(agent, env);
        
        // 4. 训练智能体（简化版）
        System.out.println("\n4. Training Agent (Behavior Cloning)...");
        VLALearningEngine learner = new BehaviorCloningLearner(0.001);
        learner.train(agent, env, 10); // 训练10个回合
        
        // 5. 评估性能
        System.out.println("\n5. Evaluating Agent Performance...");
        double avgReward = learner.evaluate(agent, env, 5);
        System.out.printf("Average Reward over 5 episodes: %.2f%n", avgReward);
        
        // 6. 清理资源
        env.close();
        
        System.out.println("\n========================================");
        System.out.println("  Demo Completed Successfully!");
        System.out.println("========================================");
    }
    
    /**
     * 运行一个演示回合
     */
    private static void runDemoEpisode(VLAAgent agent, RobotEnvironment env) {
        VLAState state = env.reset();
        double totalReward = 0.0;
        int step = 0;
        
        System.out.println("Starting episode...\n");
        
        while (step < 20) { // 最多20步
            // 智能体预测动作
            VLAAction action = agent.predict(state);
            
            // 打印动作信息
            System.out.println("Step " + step + ":");
            System.out.println("  Action Type: " + action.getActionType().getDescription());
            System.out.println("  Confidence: " + String.format("%.2f", action.getConfidence()));
            System.out.println("  Feedback: " + action.getLanguageFeedback());
            
            // 执行动作
            RobotEnvironment.EnvironmentStep envStep = env.step(action);
            
            totalReward += envStep.getReward();
            System.out.println("  Reward: " + String.format("%.2f", envStep.getReward()));
            System.out.println("  Total Reward: " + String.format("%.2f", totalReward));
            
            // 打印环境信息
            if (envStep.getInfo() != null) {
                Object grasped = envStep.getInfo().get("object_grasped");
                Object distance = envStep.getInfo().get("distance_to_target");
                System.out.println("  Object Grasped: " + grasped);
                if (distance != null) {
                    System.out.println("  Distance to Target: " + String.format("%.3f", distance));
                }
            }
            System.out.println();
            
            if (envStep.isDone()) {
                System.out.println("Episode finished!");
                break;
            }
            
            state = envStep.getNextState();
            step++;
        }
        
        System.out.println("Episode Summary:");
        System.out.println("  Total Steps: " + step);
        System.out.println("  Total Reward: " + String.format("%.2f", totalReward));
    }
}
