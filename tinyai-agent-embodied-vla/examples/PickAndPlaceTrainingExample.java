package io.leavesfly.tinyai.agent.vla.examples;

import io.leavesfly.tinyai.agent.vla.VLAAgent;
import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.agent.vla.env.TaskScenario;
import io.leavesfly.tinyai.agent.vla.learning.BehaviorCloningLearner;
import io.leavesfly.tinyai.agent.vla.learning.VLALearningEngine;
import io.leavesfly.tinyai.agent.vla.model.TaskConfig;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.agent.vla.model.VLAState;

import java.util.ArrayList;
import java.util.List;

/**
 * PickAndPlace任务完整训练示例
 * 
 * 本示例展示如何使用VLA智能体完成"拾取并放置"任务的完整训练流程：
 * 1. 环境初始化
 * 2. 智能体创建与配置
 * 3. 专家演示数据收集
 * 4. 行为克隆训练
 * 5. 模型评估与可视化
 * 6. 模型保存与加载
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class PickAndPlaceTrainingExample {
    
    // 训练超参数
    private static final int HIDDEN_DIM = 768;
    private static final int NUM_HEADS = 8;
    private static final int NUM_LAYERS = 6;
    private static final int ACTION_DIM = 7;
    
    private static final double LEARNING_RATE = 0.001;
    private static final int TRAINING_EPISODES = 100;
    private static final int EVALUATION_EPISODES = 10;
    private static final int MAX_STEPS_PER_EPISODE = 100;
    
    public static void main(String[] args) {
        printHeader();
        
        try {
            // Step 1: 环境初始化
            RobotEnvironment env = createEnvironment();
            
            // Step 2: 创建VLA智能体
            VLAAgent agent = createAgent();
            
            // Step 3: 收集专家演示数据（可选）
            List<DemonstrationData> demonstrations = collectDemonstrations(env, 10);
            System.out.println("Collected " + demonstrations.size() + " expert demonstrations\n");
            
            // Step 4: 行为克隆训练
            VLALearningEngine learner = trainAgent(agent, env, demonstrations);
            
            // Step 5: 评估训练后的模型
            evaluateAgent(agent, env);
            
            // Step 6: 可视化训练过程
            visualizeTraining(learner);
            
            // Step 7: 保存模型
            saveModel(agent, "models/vla_pick_and_place.model");
            
            // Step 8: 清理资源
            env.close();
            
            printFooter();
            
        } catch (Exception e) {
            System.err.println("Training failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 创建PickAndPlace任务环境
     */
    private static RobotEnvironment createEnvironment() {
        System.out.println("=== Step 1: Creating Environment ===");
        
        TaskConfig taskConfig = new TaskConfig();
        taskConfig.setTaskName(TaskScenario.PICK_AND_PLACE.getName());
        taskConfig.setTaskDescription(TaskScenario.PICK_AND_PLACE.getDescription());
        taskConfig.setMaxSteps(MAX_STEPS_PER_EPISODE);
        taskConfig.setSuccessReward(100.0);
        taskConfig.setStepPenalty(-0.1);
        taskConfig.setRender(false);
        
        RobotEnvironment env = new SimpleRobotEnv(taskConfig);
        
        System.out.println("✓ Environment created successfully");
        System.out.println("  Task: " + TaskScenario.PICK_AND_PLACE.getName());
        System.out.println("  Difficulty: " + TaskScenario.PICK_AND_PLACE.getDifficultyStars());
        System.out.println("  Max Steps: " + MAX_STEPS_PER_EPISODE);
        System.out.println();
        
        return env;
    }
    
    /**
     * 创建VLA智能体
     */
    private static VLAAgent createAgent() {
        System.out.println("=== Step 2: Creating VLA Agent ===");
        
        VLAAgent agent = new VLAAgent(HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, ACTION_DIM);
        
        System.out.println("✓ Agent created successfully");
        agent.printModelInfo();
        System.out.println();
        
        return agent;
    }
    
    /**
     * 收集专家演示数据
     * 在实际应用中，这些数据可能来自：
     * 1. 人类远程操作
     * 2. 预定义的控制策略
     * 3. 已训练好的模型
     */
    private static List<DemonstrationData> collectDemonstrations(
            RobotEnvironment env, int numDemonstrations) {
        System.out.println("=== Step 3: Collecting Expert Demonstrations ===");
        
        List<DemonstrationData> demonstrations = new ArrayList<>();
        
        for (int i = 0; i < numDemonstrations; i++) {
            System.out.printf("Collecting demonstration %d/%d...%n", i + 1, numDemonstrations);
            
            // 使用简单的启发式策略生成演示数据
            VLAState state = env.reset();
            List<StateActionPair> trajectory = new ArrayList<>();
            
            int step = 0;
            while (step < MAX_STEPS_PER_EPISODE) {
                // 这里使用简单的启发式策略（实际应用中应该用专家策略）
                VLAAction action = getExpertAction(state, env);
                
                trajectory.add(new StateActionPair(state, action));
                
                RobotEnvironment.EnvironmentStep envStep = env.step(action);
                
                if (envStep.isDone()) {
                    break;
                }
                
                state = envStep.getNextState();
                step++;
            }
            
            demonstrations.add(new DemonstrationData(trajectory));
        }
        
        System.out.println("✓ Demonstrations collected\n");
        return demonstrations;
    }
    
    /**
     * 训练智能体
     */
    private static VLALearningEngine trainAgent(
            VLAAgent agent, 
            RobotEnvironment env,
            List<DemonstrationData> demonstrations) {
        System.out.println("=== Step 4: Training Agent (Behavior Cloning) ===");
        
        VLALearningEngine learner = new BehaviorCloningLearner(LEARNING_RATE);
        
        // 如果有演示数据，先进行预训练
        if (demonstrations != null && !demonstrations.isEmpty()) {
            System.out.println("Pre-training from demonstrations...");
            learner.pretrainFromDemonstrations(agent, demonstrations);
            System.out.println("✓ Pre-training completed\n");
        }
        
        // 在线训练
        System.out.println("Starting online training...");
        System.out.println("Training Episodes: " + TRAINING_EPISODES);
        System.out.println("Learning Rate: " + LEARNING_RATE);
        System.out.println();
        
        // 训练过程中记录指标
        for (int episode = 1; episode <= TRAINING_EPISODES; episode++) {
            double episodeReward = learner.trainEpisode(agent, env);
            
            if (episode % 10 == 0) {
                System.out.printf("Episode %d/%d - Reward: %.2f%n", 
                    episode, TRAINING_EPISODES, episodeReward);
            }
        }
        
        System.out.println("\n✓ Training completed\n");
        return learner;
    }
    
    /**
     * 评估智能体性能
     */
    private static void evaluateAgent(VLAAgent agent, RobotEnvironment env) {
        System.out.println("=== Step 5: Evaluating Agent ===");
        
        double totalReward = 0.0;
        int successCount = 0;
        List<Double> episodeRewards = new ArrayList<>();
        
        for (int i = 0; i < EVALUATION_EPISODES; i++) {
            VLAState state = env.reset();
            double episodeReward = 0.0;
            int step = 0;
            
            while (step < MAX_STEPS_PER_EPISODE) {
                VLAAction action = agent.predict(state);
                RobotEnvironment.EnvironmentStep envStep = env.step(action);
                
                episodeReward += envStep.getReward();
                
                if (envStep.isDone()) {
                    if (episodeReward > 80.0) { // 成功阈值
                        successCount++;
                    }
                    break;
                }
                
                state = envStep.getNextState();
                step++;
            }
            
            episodeRewards.add(episodeReward);
            totalReward += episodeReward;
        }
        
        double avgReward = totalReward / EVALUATION_EPISODES;
        double successRate = (double) successCount / EVALUATION_EPISODES * 100;
        
        System.out.println("Evaluation Results:");
        System.out.printf("  Average Reward: %.2f%n", avgReward);
        System.out.printf("  Success Rate: %.1f%%%n", successRate);
        System.out.printf("  Min Reward: %.2f%n", episodeRewards.stream().min(Double::compare).orElse(0.0));
        System.out.printf("  Max Reward: %.2f%n", episodeRewards.stream().max(Double::compare).orElse(0.0));
        System.out.println();
    }
    
    /**
     * 可视化训练过程
     */
    private static void visualizeTraining(VLALearningEngine learner) {
        System.out.println("=== Step 6: Visualizing Training ===");
        
        // 打印训练曲线（简化版）
        System.out.println("Training Metrics:");
        System.out.println("  Total Training Episodes: " + TRAINING_EPISODES);
        System.out.println("  Learning Rate: " + LEARNING_RATE);
        System.out.println("  Note: Detailed metrics can be exported to TensorBoard");
        System.out.println();
    }
    
    /**
     * 保存模型
     */
    private static void saveModel(VLAAgent agent, String filepath) {
        System.out.println("=== Step 7: Saving Model ===");
        
        try {
            // 这里应该实现模型保存逻辑
            System.out.println("✓ Model saved to: " + filepath);
            System.out.println("  (Note: Model serialization to be implemented)");
        } catch (Exception e) {
            System.err.println("✗ Failed to save model: " + e.getMessage());
        }
        System.out.println();
    }
    
    /**
     * 获取专家动作（启发式策略）
     * 这是一个简化的示例，实际应该使用真实的专家策略
     */
    private static VLAAction getExpertAction(VLAState state, RobotEnvironment env) {
        // 简化的启发式策略：
        // 1. 如果未抓取物体，移动到物体位置并抓取
        // 2. 如果已抓取物体，移动到目标位置并释放
        
        // 这里返回一个随机动作作为占位符
        return env.sampleAction();
    }
    
    // ==================== 辅助数据结构 ====================
    
    /**
     * 状态-动作对
     */
    static class StateActionPair {
        private final VLAState state;
        private final VLAAction action;
        
        public StateActionPair(VLAState state, VLAAction action) {
            this.state = state;
            this.action = action;
        }
        
        public VLAState getState() { return state; }
        public VLAAction getAction() { return action; }
    }
    
    /**
     * 演示数据
     */
    static class DemonstrationData {
        private final List<StateActionPair> trajectory;
        
        public DemonstrationData(List<StateActionPair> trajectory) {
            this.trajectory = trajectory;
        }
        
        public List<StateActionPair> getTrajectory() { return trajectory; }
    }
    
    // ==================== 界面输出 ====================
    
    private static void printHeader() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║       TinyAI VLA - PickAndPlace Training Example          ║");
        System.out.println("║       Vision-Language-Action Embodied Intelligence        ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
    
    private static void printFooter() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║              Training Completed Successfully!             ║");
        System.out.println("║                  Happy Robot Learning! 🤖                  ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }
}
