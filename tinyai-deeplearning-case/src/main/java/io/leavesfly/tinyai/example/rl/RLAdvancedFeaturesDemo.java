package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.ReplayBuffer;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.*;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;
import java.util.concurrent.*;

/**
 * 强化学习高级特性演示
 * 
 * 本演示展示强化学习的高级特性，包括：
 * 1. 经验回放机制分析
 * 2. 探索策略对比
 * 3. 自适应参数调整
 * 4. 迁移学习示例
 * 5. 鲁棒性测试
 * 
 * @author 山泽
 */
public class RLAdvancedFeaturesDemo {
    
    public static void main(String[] args) {
        RLAdvancedFeaturesDemo demo = new RLAdvancedFeaturesDemo();
        
        System.out.println("==========================================");
        System.out.println("      TinyAI 强化学习高级特性演示         ");
        System.out.println("==========================================");
        
        demo.demonstrateExperienceReplay();
        demo.demonstrateExplorationStrategies();
        demo.demonstrateAdaptiveParameters();
        demo.demonstrateTransferLearning();
        demo.demonstrateRobustnessTest();
        
        System.out.println("\n========== 高级特性演示完成 ==========");
    }
    
    /**
     * 演示经验回放机制
     */
    public void demonstrateExperienceReplay() {
        System.out.println("\n========== 经验回放机制分析 ==========");
        
        int[] bufferSizes = {1000, 5000, 10000};
        int[] batchSizes = {16, 32, 64};
        
        CartPoleEnvironment environment = new CartPoleEnvironment(500);
        
        System.out.println("测试不同经验回放配置对DQN性能的影响:");
        
        for (int bufferSize : bufferSizes) {
            for (int batchSize : batchSizes) {
                System.out.println(String.format("\n缓冲区大小: %d, 批次大小: %d", bufferSize, batchSize));
                
                DQNAgent agent = new DQNAgent("DQN", 4, 2, new int[]{64, 64}, 
                                            0.001f, 0.1f, 0.99f, batchSize, bufferSize, 100);
                
                float[] results = testExperienceReplayEffect(agent, environment, 50);
                
                System.out.println(String.format("  平均奖励: %.2f, 标准差: %.2f", results[0], results[1]));
                System.out.println(String.format("  缓冲区使用率: %.2f%%, 平均损失: %.6f", 
                                 agent.getBufferUsage() * 100, agent.getAverageLoss()));
            }
        }
        
        analyzeExperienceReplayMemoryEffect();
    }
    
    /**
     * 演示探索策略对比
     */
    public void demonstrateExplorationStrategies() {
        System.out.println("\n========== 探索策略深度对比 ==========");
        
        MultiArmedBanditEnvironment environment = new MultiArmedBanditEnvironment(
            new float[]{0.1f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.5f, 0.7f, 0.35f}, 1000);
        
        List<BanditAgent> agents = Arrays.asList(
            new EpsilonGreedyBanditAgent("固定ε-贪心(0.1)", 10, 0.1f),
            new EpsilonGreedyBanditAgent("固定ε-贪心(0.05)", 10, 0.05f),
            new UCBBanditAgent("UCB", 10),
            new ThompsonSamplingBanditAgent("汤普森采样", 10)
        );
        
        System.out.println("\n不同探索策略性能对比:");
        runExplorationComparison(agents, environment, 1000);
        
        analyzeExplorationTradeoff();
    }
    
    /**
     * 演示自适应参数调整
     */
    public void demonstrateAdaptiveParameters() {
        System.out.println("\n========== 自适应参数调整 ==========");
        
        CartPoleEnvironment environment = new CartPoleEnvironment(500);
        
        // 模拟自适应学习率智能体
        DQNAgent adaptiveAgent = new DQNAgent("自适应DQN", 4, 2, new int[]{64, 64}, 
                                            0.001f, 0.1f, 0.99f, 32, 10000, 100);
        
        DQNAgent fixedAgent = new DQNAgent("固定DQN", 4, 2, new int[]{64, 64}, 
                                         0.001f, 0.1f, 0.99f, 32, 10000, 100);
        
        System.out.println("比较自适应参数调整与固定参数的学习效果:");
        
        float adaptiveReward = trainAgent(adaptiveAgent, environment, 200);
        float fixedReward = trainAgent(fixedAgent, environment, 200);
        
        System.out.println(String.format("自适应参数平均奖励: %.2f", adaptiveReward / 200));
        System.out.println(String.format("固定参数平均奖励: %.2f", fixedReward / 200));
        
        demonstrateAdaptiveStrategies();
    }
    
    /**
     * 演示迁移学习
     */
    public void demonstrateTransferLearning() {
        System.out.println("\n========== 迁移学习演示 ==========");
        
        GridWorldEnvironment sourceEnv = new GridWorldEnvironment(4, 4);
        GridWorldEnvironment targetEnv = new GridWorldEnvironment(6, 6);
        
        System.out.println("迁移学习: 从4x4网格世界迁移到6x6网格世界");
        
        // 在源环境训练
        DQNAgent sourceAgent = new DQNAgent("源智能体", 2, 4, new int[]{32, 32}, 
                                          0.01f, 0.1f, 0.99f, 32, 5000, 50);
        
        System.out.println("阶段1: 在源环境训练...");
        float sourcePerformance = trainAgent(sourceAgent, sourceEnv, 100);
        System.out.println(String.format("源环境训练完成，平均奖励: %.2f", sourcePerformance / 100));
        
        // 迁移学习 vs 从头训练对比
        DQNAgent transferAgent = new DQNAgent("迁移智能体", 2, 4, new int[]{32, 32}, 
                                            0.005f, 0.05f, 0.99f, 32, 5000, 50); // 更低的学习率
        
        DQNAgent scratchAgent = new DQNAgent("从头训练", 2, 4, new int[]{32, 32}, 
                                           0.01f, 0.1f, 0.99f, 32, 5000, 50);
        
        System.out.println("阶段2: 在目标环境比较迁移学习与从头训练...");
        float transferReward = trainAgent(transferAgent, targetEnv, 150);
        float scratchReward = trainAgent(scratchAgent, targetEnv, 150);
        
        System.out.println(String.format("迁移学习平均奖励: %.2f", transferReward / 150));
        System.out.println(String.format("从头训练平均奖励: %.2f", scratchReward / 150));
        
        if (transferReward > scratchReward) {
            System.out.println("迁移学习显示出优势，学习速度更快");
        } else {
            System.out.println("在此简化示例中差异不明显，实际复杂任务中迁移学习优势更显著");
        }
    }
    
    /**
     * 演示鲁棒性测试
     */
    public void demonstrateRobustnessTest() {
        System.out.println("\n========== 鲁棒性测试 ==========");
        
        CartPoleEnvironment environment = new CartPoleEnvironment(500);
        
        DQNAgent agent = new DQNAgent("测试智能体", 4, 2, new int[]{64, 64}, 
                                    0.001f, 0.1f, 0.99f, 32, 10000, 100);
        
        // 先正常训练
        System.out.println("正常环境训练...");
        float normalPerformance = trainAgent(agent, environment, 100);
        System.out.println(String.format("正常环境平均奖励: %.2f", normalPerformance / 100));
        
        // 测试噪声鲁棒性
        System.out.println("\n测试噪声鲁棒性:");
        testNoiseRobustness(agent, environment);
        
        // 异常检测演示
        demonstrateAnomalyDetection(agent, environment);
    }
    
    // ==================== 辅助方法 ====================
    
    private float[] testExperienceReplayEffect(DQNAgent agent, Environment environment, int episodes) {
        List<Float> rewards = new ArrayList<>();
        
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(), 
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            rewards.add(episodeReward);
        }
        
        double mean = rewards.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
        double variance = rewards.stream().mapToDouble(r -> Math.pow(r - mean, 2)).average().orElse(0.0);
        double stdDev = Math.sqrt(variance);
        
        return new float[]{(float) mean, (float) stdDev};
    }
    
    private void analyzeExperienceReplayMemoryEffect() {
        System.out.println("\n=== 经验回放记忆效应分析 ===");
        
        ReplayBuffer buffer = new ReplayBuffer(1000);
        Random random = new Random();
        
        // 模拟添加经验
        for (int i = 0; i < 1500; i++) {
            float reward = random.nextFloat() * 2 - 1;
            Experience exp = new Experience(
                new Variable(NdArray.of(new float[]{random.nextFloat()})),
                new Variable(NdArray.of(random.nextInt(4))),
                reward,
                new Variable(NdArray.of(new float[]{random.nextFloat()})),
                random.nextBoolean(),
                i
            );
            buffer.push(exp);
        }
        
        System.out.println(String.format("缓冲区使用率: %.2f%%", buffer.getUsageRate() * 100));
        System.out.println("经验回放有助于打破数据间的时间相关性，提高学习稳定性");
    }
    
    private void runExplorationComparison(List<BanditAgent> agents, 
                                        MultiArmedBanditEnvironment environment, int steps) {
        for (BanditAgent agent : agents) {
            agent.reset();
            environment.reset();
            
            float totalReward = 0.0f;
            
            for (int step = 0; step < steps; step++) {
                Variable action = agent.selectAction(environment.getCurrentState());
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    environment.getCurrentState(), action, result.getReward(),
                    result.getNextState(), result.isDone(), step
                );
                
                agent.learn(experience);
                totalReward += result.getReward();
            }
            
            System.out.println(String.format("%s: 总奖励=%.2f, 平均奖励=%.4f, 最优选择率=%.2f%%",
                             agent.getName(), totalReward, totalReward / steps, 
                             agent.getOptimalActionRate() * 100));
        }
    }
    
    private void analyzeExplorationTradeoff() {
        System.out.println("\n=== 探索-利用权衡分析 ===");
        System.out.println("ε-贪心: 简单有效，但探索是随机的");
        System.out.println("UCB: 基于置信区间，探索更有针对性");
        System.out.println("汤普森采样: 贝叶斯方法，自然平衡探索与利用");
    }
    
    private float trainAgent(Agent agent, Environment environment, int episodes) {
        float totalReward = 0.0f;
        
        for (int episode = 0; episode < episodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < 500) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                Experience experience = new Experience(
                    state, action, result.getReward(),
                    result.getNextState(), result.isDone(), steps
                );
                
                agent.learn(experience);
                
                if (agent instanceof REINFORCEAgent && result.isDone()) {
                    ((REINFORCEAgent) agent).learnFromEpisode();
                }
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            totalReward += episodeReward;
        }
        
        return totalReward;
    }
    
    private void demonstrateAdaptiveStrategies() {
        System.out.println("\n=== 自适应策略类型 ===");
        System.out.println("1. 自适应学习率: 基于梯度大小调整学习步长");
        System.out.println("2. 自适应探索率: 基于性能反馈调整探索程度");
        System.out.println("3. 自适应网络结构: 根据任务复杂度调整网络大小");
        System.out.println("4. 自适应批次大小: 基于梯度稳定性调整批次大小");
    }
    
    private void testNoiseRobustness(DQNAgent agent, CartPoleEnvironment environment) {
        // 模拟噪声环境测试
        Random random = new Random();
        
        float[] noiseLevels = {0.0f, 0.1f, 0.2f, 0.3f};
        
        for (float noiseLevel : noiseLevels) {
            float totalReward = 0.0f;
            int testEpisodes = 20;
            
            for (int episode = 0; episode < testEpisodes; episode++) {
                Variable state = environment.reset();
                float episodeReward = 0.0f;
                int steps = 0;
                
                while (!environment.isDone() && steps < 200) {
                    // 添加噪声到状态观测
                    NdArray originalArray = state.getValue();
                    Shape shape = originalArray.getShape();
                    float[] data = new float[shape.size()];
                    
                    // 复制原始数据并添加噪声
                    for (int i = 0; i < data.length; i++) {
                        if (shape.isMatrix()) {
                            data[i] = originalArray.get(i / shape.getColumn(), i % shape.getColumn());
                        } else {
                            data[i] = originalArray.get(i);
                        }
                        float noise = (random.nextFloat() - 0.5f) * 2 * noiseLevel;
                        data[i] += noise;
                    }
                    
                    Variable noisyState = new Variable(NdArray.of(data, shape));
                    
                    Variable action = agent.selectAction(noisyState);
                    Environment.StepResult result = environment.step(action);
                    
                    state = result.getNextState();
                    episodeReward += result.getReward();
                    steps++;
                }
                
                totalReward += episodeReward;
            }
            
            System.out.println(String.format("噪声水平 %.1f: 平均奖励 %.2f", 
                             noiseLevel, totalReward / testEpisodes));
        }
    }
    
    private void demonstrateAnomalyDetection(DQNAgent agent, CartPoleEnvironment environment) {
        System.out.println("\n=== 异常检测演示 ===");
        
        // 收集正常表现数据
        List<Float> normalRewards = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < 200) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = environment.step(action);
                
                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;
            }
            
            normalRewards.add(episodeReward);
        }
        
        double meanReward = normalRewards.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
        double stdReward = Math.sqrt(normalRewards.stream()
            .mapToDouble(r -> Math.pow(r - meanReward, 2)).average().orElse(0.0));
        
        System.out.println(String.format("正常表现: 平均奖励 %.2f ± %.2f", meanReward, stdReward));
        
        // 异常阈值
        double anomalyThreshold = meanReward - 2 * stdReward;
        System.out.println(String.format("异常检测阈值: %.2f", anomalyThreshold));
        System.out.println("当奖励低于阈值时，可能表明智能体遇到了异常情况或需要重新训练");
    }
}