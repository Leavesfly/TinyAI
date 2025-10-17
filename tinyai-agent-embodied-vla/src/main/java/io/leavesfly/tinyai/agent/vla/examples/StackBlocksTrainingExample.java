package io.leavesfly.tinyai.agent.vla.examples;

import io.leavesfly.tinyai.agent.vla.VLAAgent;
import io.leavesfly.tinyai.agent.vla.env.RobotEnvironment;
import io.leavesfly.tinyai.agent.vla.env.SimpleRobotEnv;
import io.leavesfly.tinyai.agent.vla.env.TaskScenario;
import io.leavesfly.tinyai.agent.vla.learning.BehaviorCloningLearner;
import io.leavesfly.tinyai.agent.vla.model.TaskConfig;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.agent.vla.model.VLAState;

/**
 * StackBlocks任务训练示例
 * 
 * 堆叠方块任务是比PickAndPlace更复杂的任务，需要：
 * 1. 精确的视觉感知（判断方块位置和稳定性）
 * 2. 序列化决策（先放底层，再放上层）
 * 3. 细致的力控制（避免推倒已堆叠的方块）
 * 
 * 本示例展示如何使用课程学习（Curriculum Learning）策略
 * 从简单到复杂逐步训练智能体。
 * 
 * @author TinyAI Team
 * @version 1.0
 */
public class StackBlocksTrainingExample {
    
    // 模型配置
    private static final int HIDDEN_DIM = 768;
    private static final int NUM_HEADS = 8;
    private static final int NUM_LAYERS = 6;
    private static final int ACTION_DIM = 7;
    
    // 课程学习配置
    private static final int[] CURRICULUM_BLOCKS = {2, 3, 4}; // 从2个方块开始，逐渐增加到4个
    private static final int[] CURRICULUM_EPISODES = {30, 50, 70}; // 每个阶段的训练回合数
    
    public static void main(String[] args) {
        printHeader();
        
        try {
            // 创建智能体
            VLAAgent agent = createAgent();
            
            // 课程学习训练
            curriculumLearning(agent);
            
            // 最终评估
            finalEvaluation(agent);
            
            printFooter();
            
        } catch (Exception e) {
            System.err.println("Training failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 创建VLA智能体
     */
    private static VLAAgent createAgent() {
        System.out.println("=== Creating VLA Agent ===");
        
        VLAAgent agent = new VLAAgent(HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, ACTION_DIM);
        
        System.out.println("✓ Agent created");
        agent.printModelInfo();
        System.out.println();
        
        return agent;
    }
    
    /**
     * 课程学习训练
     * 逐步增加任务难度，帮助模型更好地学习
     */
    private static void curriculumLearning(VLAAgent agent) {
        System.out.println("=== Curriculum Learning ===");
        System.out.println("Training Strategy: Start simple, then increase difficulty\n");
        
        for (int stage = 0; stage < CURRICULUM_BLOCKS.length; stage++) {
            int numBlocks = CURRICULUM_BLOCKS[stage];
            int episodes = CURRICULUM_EPISODES[stage];
            
            System.out.println("┌─────────────────────────────────────────┐");
            System.out.printf("│ Stage %d: Stacking %d Blocks            │%n", stage + 1, numBlocks);
            System.out.println("└─────────────────────────────────────────┘");
            
            // 创建对应难度的环境
            RobotEnvironment env = createEnvironment(numBlocks);
            
            // 训练该阶段
            trainStage(agent, env, episodes, numBlocks);
            
            // 评估该阶段
            evaluateStage(agent, env, numBlocks);
            
            env.close();
            System.out.println();
        }
        
        System.out.println("✓ Curriculum Learning Completed\n");
    }
    
    /**
     * 创建指定难度的环境
     */
    private static RobotEnvironment createEnvironment(int numBlocks) {
        TaskConfig taskConfig = new TaskConfig();
        taskConfig.setTaskName("Stack " + numBlocks + " Blocks");
        taskConfig.setTaskDescription(TaskScenario.STACK_BLOCKS.getDescription());
        taskConfig.setMaxSteps(100 + numBlocks * 20); // 更多方块需要更多步数
        taskConfig.setSuccessReward(100.0 * numBlocks); // 奖励随难度增加
//        taskConfig.setStepPenalty(-0.1);
        taskConfig.setRender(false);
        
        // 设置特定参数
//        taskConfig.addParameter("num_blocks", numBlocks);
        
        return new SimpleRobotEnv(taskConfig);
    }
    
    /**
     * 训练某个阶段
     */
    private static void trainStage(VLAAgent agent, RobotEnvironment env, 
                                    int episodes, int numBlocks) {
        System.out.println("Training Phase:");
        System.out.println("  Episodes: " + episodes);
        System.out.println("  Learning Rate: 0.001");
        System.out.println();
        
        BehaviorCloningLearner learner = new BehaviorCloningLearner(0.001);
        
        double totalReward = 0.0;
        int successCount = 0;
        
        for (int episode = 1; episode <= episodes; episode++) {
            double episodeReward = learner.trainEpisode(agent, env);
            totalReward += episodeReward;
            
            if (episodeReward > 80.0 * numBlocks) {
                successCount++;
            }
            
            if (episode % 10 == 0) {
                double avgReward = totalReward / episode;
                double successRate = (double) successCount / episode * 100;
                
                System.out.printf("  Episode %3d - Avg Reward: %6.2f | Success Rate: %5.1f%%%n",
                    episode, avgReward, successRate);
            }
        }
        
        System.out.println();
    }
    
    /**
     * 评估阶段性能
     */
    private static void evaluateStage(VLAAgent agent, RobotEnvironment env, int numBlocks) {
        System.out.println("Stage Evaluation:");
        
        int evalEpisodes = 10;
        double totalReward = 0.0;
        int successCount = 0;
        int perfectStackCount = 0;
        
        for (int i = 0; i < evalEpisodes; i++) {
            VLAState state = env.reset();
            double episodeReward = 0.0;
            int blocksStacked = 0;
            
            for (int step = 0; step < 200; step++) {
                VLAAction action = agent.predict(state);
                RobotEnvironment.EnvironmentStep envStep = env.step(action);
                
                episodeReward += envStep.getReward();
                
                // 检查堆叠进度
                if (envStep.getInfo() != null) {
                    Object stacked = envStep.getInfo().get("blocks_stacked");
                    if (stacked instanceof Integer) {
                        blocksStacked = Math.max(blocksStacked, (Integer) stacked);
                    }
                }
                
                if (envStep.isDone()) {
                    break;
                }
                
                state = envStep.getNextState();
            }
            
            totalReward += episodeReward;
            
            if (blocksStacked >= numBlocks - 1) {
                successCount++;
            }
            if (blocksStacked == numBlocks) {
                perfectStackCount++;
            }
        }
        
        double avgReward = totalReward / evalEpisodes;
        double successRate = (double) successCount / evalEpisodes * 100;
        double perfectRate = (double) perfectStackCount / evalEpisodes * 100;
        
        System.out.printf("  Average Reward: %.2f%n", avgReward);
        System.out.printf("  Success Rate: %.1f%% (>=%d blocks)%n", successRate, numBlocks - 1);
        System.out.printf("  Perfect Rate: %.1f%% (all %d blocks)%n", perfectRate, numBlocks);
        System.out.println();
    }
    
    /**
     * 最终综合评估
     */
    private static void finalEvaluation(VLAAgent agent) {
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║        Final Evaluation                ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println();
        
        // 在所有难度下测试
        for (int numBlocks : CURRICULUM_BLOCKS) {
            System.out.println("Testing with " + numBlocks + " blocks:");
            
            RobotEnvironment env = createEnvironment(numBlocks);
            
            double totalReward = 0.0;
            int successCount = 0;
            
            for (int i = 0; i < 20; i++) {
                VLAState state = env.reset();
                double episodeReward = 0.0;
                
                for (int step = 0; step < 200; step++) {
                    VLAAction action = agent.predict(state);
                    RobotEnvironment.EnvironmentStep envStep = env.step(action);
                    
                    episodeReward += envStep.getReward();
                    
                    if (envStep.isDone()) {
                        break;
                    }
                    
                    state = envStep.getNextState();
                }
                
                totalReward += episodeReward;
                
                if (episodeReward > 80.0 * numBlocks) {
                    successCount++;
                }
            }
            
            double avgReward = totalReward / 20;
            double successRate = (double) successCount / 20 * 100;
            
            System.out.printf("  Average Reward: %.2f | Success Rate: %.1f%%%n", 
                avgReward, successRate);
            
            env.close();
        }
        
        System.out.println();
    }
    
    // ==================== 界面输出 ====================
    
    private static void printHeader() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║       TinyAI VLA - StackBlocks Training Example           ║");
        System.out.println("║          Curriculum Learning for Complex Tasks            ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();
    }
    
    private static void printFooter() {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║              Training Completed Successfully!             ║");
        System.out.println("║         Your agent can now stack blocks! 📦📦📦            ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
    }
}
