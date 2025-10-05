package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Agent;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.*;

import java.util.*;
import java.text.DecimalFormat;

/**
 * 强化学习算法性能分析演示
 * 
 * 本演示提供深度的性能分析功能，包括：
 * 1. 多算法批量性能对比
 * 2. 学习曲线分析
 * 3. 收敛性分析
 * 4. 参数敏感性分析
 * 5. 统计显著性检验
 * 6. 性能报告生成
 * 
 * @author 山泽
 */
public class RLPerformanceAnalysisDemo {
    
    private static final DecimalFormat df = new DecimalFormat("#.####");
    
    // 测试配置
    private static class TestConfig {
        int episodes = 100;
        int runs = 10;
        int maxStepsPerEpisode = 1000;
        boolean enableLearningCurve = true;
        boolean enableConvergenceAnalysis = true;
        boolean enableStatisticalTest = true;
        
        public TestConfig(int episodes, int runs) {
            this.episodes = episodes;
            this.runs = runs;
        }
    }
    
    // 性能指标
    private static class PerformanceMetrics {
        String algorithmName;
        List<Float> episodeRewards = new ArrayList<>();
        List<Float> cumulativeRewards = new ArrayList<>();
        List<Integer> convergencePoints = new ArrayList<>();
        double meanReward;
        double stdReward;
        double medianReward;
        double maxReward;
        double minReward;
        int convergenceEpisode = -1;
        double convergenceThreshold = 0.01;
        
        public PerformanceMetrics(String algorithmName) {
            this.algorithmName = algorithmName;
        }
        
        public void computeStatistics() {
            if (episodeRewards.isEmpty()) return;
            
            // 基础统计
            meanReward = episodeRewards.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
            
            double variance = episodeRewards.stream()
                .mapToDouble(r -> Math.pow(r - meanReward, 2))
                .average().orElse(0.0);
            stdReward = Math.sqrt(variance);
            
            List<Float> sorted = new ArrayList<>(episodeRewards);
            Collections.sort(sorted);
            medianReward = sorted.get(sorted.size() / 2);
            maxReward = Collections.max(episodeRewards);
            minReward = Collections.min(episodeRewards);
            
            // 收敛性分析
            analyzeConvergence();
        }
        
        private void analyzeConvergence() {
            if (episodeRewards.size() < 20) return;
            
            int windowSize = Math.max(10, episodeRewards.size() / 10);
            
            for (int i = windowSize; i < episodeRewards.size() - windowSize; i++) {
                double windowMean = episodeRewards.subList(i - windowSize / 2, i + windowSize / 2)
                    .stream().mapToDouble(Double::valueOf).average().orElse(0.0);
                
                double windowStd = Math.sqrt(episodeRewards.subList(i - windowSize / 2, i + windowSize / 2)
                    .stream().mapToDouble(r -> Math.pow(r - windowMean, 2)).average().orElse(0.0));
                
                if (windowStd / Math.abs(windowMean) < convergenceThreshold && convergenceEpisode == -1) {
                    convergenceEpisode = i;
                    break;
                }
            }
        }
    }
    
    public static void main(String[] args) {
        RLPerformanceAnalysisDemo demo = new RLPerformanceAnalysisDemo();
        
        System.out.println("=======================================");
        System.out.println("    TinyAI 强化学习性能分析系统       ");
        System.out.println("=======================================");
        
        // 运行不同环境的性能分析
        demo.runMultiArmedBanditAnalysis();
        demo.runCartPoleAnalysis();
        demo.runGridWorldAnalysis();
        
        // 生成综合报告
        demo.generateComprehensiveReport();
    }
    
    /**
     * 多臂老虎机环境性能分析
     */
    public void runMultiArmedBanditAnalysis() {
        System.out.println("\n========== 多臂老虎机环境性能分析 ==========");
        
        // 测试配置
        TestConfig config = new TestConfig(1000, 20);
        
        // 创建测试环境
        float[] rewards = {0.1f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.5f, 0.7f, 0.35f};
        MultiArmedBanditEnvironment environment = new MultiArmedBanditEnvironment(rewards, config.maxStepsPerEpisode);
        
        // 测试算法列表
        List<AgentFactory> agentFactories = Arrays.asList(
            () -> new EpsilonGreedyBanditAgent("ε-贪心(0.1)", 10, 0.1f),
            () -> new EpsilonGreedyBanditAgent("ε-贪心(0.05)", 10, 0.05f),
            () -> new UCBBanditAgent("UCB", 10),
            () -> new ThompsonSamplingBanditAgent("汤普森采样", 10)
        );
        
        // 运行性能分析
        List<PerformanceMetrics> results = runPerformanceAnalysis(agentFactories, environment, config);
        
        // 显示分析结果
        displayBanditAnalysisResults(results);
        
        // 悔恨分析
        performRegretAnalysis(agentFactories, environment, config);
    }
    
    /**
     * CartPole环境性能分析
     */
    public void runCartPoleAnalysis() {
        System.out.println("\n========== CartPole环境性能分析 ==========");
        
        TestConfig config = new TestConfig(200, 10);
        CartPoleEnvironment environment = new CartPoleEnvironment(500);
        
        List<AgentFactory> agentFactories = Arrays.asList(
            () -> new DQNAgent("DQN", 4, 2, new int[]{64, 64}, 0.001f, 0.1f, 0.99f, 32, 10000, 100),
            () -> new REINFORCEAgent("REINFORCE", 4, 2, new int[]{64, 64}, 0.001f, 0.99f, false),
            () -> new REINFORCEAgent("REINFORCE+基线", 4, 2, new int[]{64, 64}, 0.001f, 0.99f, true)
        );
        
        List<PerformanceMetrics> results = runPerformanceAnalysis(agentFactories, environment, config);
        displayDeepRLAnalysisResults(results);
        
        // 学习曲线分析
        performLearningCurveAnalysis(results);
    }
    
    /**
     * GridWorld环境性能分析
     */
    public void runGridWorldAnalysis() {
        System.out.println("\n========== GridWorld环境性能分析 ==========");
        
        TestConfig config = new TestConfig(300, 8);
        GridWorldEnvironment environment = new GridWorldEnvironment(5, 5);
        
        List<AgentFactory> agentFactories = Arrays.asList(
            () -> new DQNAgent("DQN", 2, 4, new int[]{32, 32}, 0.01f, 0.1f, 0.99f, 32, 5000, 50),
            () -> new REINFORCEAgent("REINFORCE", 2, 4, new int[]{32, 32}, 0.01f, 0.99f, true)
        );
        
        List<PerformanceMetrics> results = runPerformanceAnalysis(agentFactories, environment, config);
        displayDeepRLAnalysisResults(results);
    }
    
    /**
     * 运行性能分析
     */
    private List<PerformanceMetrics> runPerformanceAnalysis(List<AgentFactory> agentFactories, 
                                                           Environment environment, TestConfig config) {
        List<PerformanceMetrics> allResults = new ArrayList<>();
        
        for (AgentFactory factory : agentFactories) {
            Agent agent = factory.create();
            System.out.println("\n测试算法: " + agent.getName());
            
            PerformanceMetrics metrics = new PerformanceMetrics(agent.getName());
            
            // 多次独立运行
            for (int run = 0; run < config.runs; run++) {
                System.out.print("运行 " + (run + 1) + "/" + config.runs + " ");
                
                List<Float> episodeRewards = runSingleRun(agent, environment, config);
                metrics.episodeRewards.addAll(episodeRewards);
                
                if (run % 5 == 4) System.out.println();
            }
            
            if (config.runs % 5 != 0) System.out.println();
            
            metrics.computeStatistics();
            allResults.add(metrics);
        }
        
        return allResults;
    }
    
    /**
     * 运行单次实验
     */
    private List<Float> runSingleRun(Agent agent, Environment environment, TestConfig config) {
        // 重置智能体
        agent.reset();
        if (agent instanceof BanditAgent) {
            ((BanditAgent) agent).reset();
        }
        
        List<Float> episodeRewards = new ArrayList<>();
        
        for (int episode = 0; episode < config.episodes; episode++) {
            Variable state = environment.reset();
            float episodeReward = 0.0f;
            int steps = 0;
            
            while (!environment.isDone() && steps < config.maxStepsPerEpisode) {
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
            
            // REINFORCE需要在回合结束时学习
            if (agent instanceof REINFORCEAgent) {
                ((REINFORCEAgent) agent).learnFromEpisode();
            }
            
            episodeRewards.add(episodeReward);
        }
        
        return episodeRewards;
    }
    
    /**
     * 显示多臂老虎机分析结果
     */
    private void displayBanditAnalysisResults(List<PerformanceMetrics> results) {
        System.out.println("\n=== 多臂老虎机性能分析结果 ===");
        System.out.printf("%-20s | %10s | %10s | %10s | %12s%n", 
                         "算法", "平均奖励", "标准差", "中位数", "最优选择率");
        System.out.println("-----|----------|----------|----------|------------");
        
        // Sort by mean reward
        results.sort((a, b) -> Double.compare(b.meanReward, a.meanReward));
        
        for (PerformanceMetrics metrics : results) {
            double optimalRate = calculateOptimalActionRate(metrics);
            System.out.printf("%-20s | %10s | %10s | %10s | %11s%%%n",
                             metrics.algorithmName,
                             df.format(metrics.meanReward),
                             df.format(metrics.stdReward),
                             df.format(metrics.medianReward),
                             df.format(optimalRate * 100));
        }
        
        performStatisticalSignificanceTest(results);
    }
    
    /**
     * 显示深度强化学习分析结果
     */
    private void displayDeepRLAnalysisResults(List<PerformanceMetrics> results) {
        System.out.println("\n=== 深度强化学习性能分析结果 ===");
        System.out.printf("%-20s | %10s | %10s | %10s | %10s | %10s%n", 
                         "算法", "平均奖励", "标准差", "最大奖励", "最小奖励", "收敛回合");
        System.out.println("-----|----------|----------|----------|----------|----------");
        
        results.sort((a, b) -> Double.compare(b.meanReward, a.meanReward));
        
        for (PerformanceMetrics metrics : results) {
            String convergence = metrics.convergenceEpisode == -1 ? "未收敛" : String.valueOf(metrics.convergenceEpisode);
            System.out.printf("%-20s | %10s | %10s | %10s | %10s | %10s%n",
                             metrics.algorithmName,
                             df.format(metrics.meanReward),
                             df.format(metrics.stdReward),
                             df.format(metrics.maxReward),
                             df.format(metrics.minReward),
                             convergence);
        }
        
        performStatisticalSignificanceTest(results);
    }
    
    /**
     * 计算最优动作选择率（仅适用于多臂老虎机）
     */
    private double calculateOptimalActionRate(PerformanceMetrics metrics) {
        // 简化计算，基于平均奖励估算
        double maxPossibleReward = 0.9; // 最优臂奖励
        return Math.min(1.0, metrics.meanReward / maxPossibleReward);
    }
    
    /**
     * 统计显著性检验
     */
    private void performStatisticalSignificanceTest(List<PerformanceMetrics> results) {
        if (results.size() < 2) return;
        
        System.out.println("\n=== 统计显著性检验 (T-Test) ===");
        
        for (int i = 0; i < results.size() - 1; i++) {
            for (int j = i + 1; j < results.size(); j++) {
                PerformanceMetrics metrics1 = results.get(i);
                PerformanceMetrics metrics2 = results.get(j);
                
                double tStatistic = calculateTStatistic(metrics1, metrics2);
                double pValue = calculatePValue(tStatistic, metrics1.episodeRewards.size() + metrics2.episodeRewards.size() - 2);
                
                String significance = pValue < 0.05 ? "显著" : "不显著";
                System.out.printf("%s vs %s: t=%.4f, p=%.4f (%s)%n",
                                 metrics1.algorithmName, metrics2.algorithmName,
                                 tStatistic, pValue, significance);
            }
        }
    }
    
    /**
     * 计算t统计量
     */
    private double calculateTStatistic(PerformanceMetrics m1, PerformanceMetrics m2) {
        double pooledStd = Math.sqrt(((m1.episodeRewards.size() - 1) * m1.stdReward * m1.stdReward +
                                     (m2.episodeRewards.size() - 1) * m2.stdReward * m2.stdReward) /
                                    (m1.episodeRewards.size() + m2.episodeRewards.size() - 2));
        
        double standardError = pooledStd * Math.sqrt(1.0 / m1.episodeRewards.size() + 1.0 / m2.episodeRewards.size());
        
        return (m1.meanReward - m2.meanReward) / standardError;
    }
    
    /**
     * 计算p值（简化版）
     */
    private double calculatePValue(double tStatistic, int degreesOfFreedom) {
        // 简化的p值计算，实际应该使用t分布
        double absT = Math.abs(tStatistic);
        if (absT > 2.576) return 0.01; // 99% 置信度
        if (absT > 1.96) return 0.05;  // 95% 置信度
        if (absT > 1.645) return 0.10; // 90% 置信度
        return 0.20; // 不显著
    }
    
    /**
     * 悔恨分析（多臂老虎机专用）
     */
    private void performRegretAnalysis(List<AgentFactory> agentFactories, 
                                     MultiArmedBanditEnvironment environment, TestConfig config) {
        System.out.println("\n=== 悔恨分析 ===");
        
        // 计算理论最优奖励
        float maxReward = 0.9f; // 已知最优臂奖励
        
        for (AgentFactory factory : agentFactories) {
            Agent agent = factory.create();
            System.out.println("\n算法: " + agent.getName());
            
            List<Float> cumulativeRegrets = new ArrayList<>();
            
            for (int run = 0; run < Math.min(5, config.runs); run++) {
                agent.reset();
                if (agent instanceof BanditAgent) {
                    ((BanditAgent) agent).reset();
                }
                
                environment.reset();
                float cumulativeRegret = 0.0f;
                
                for (int step = 0; step < config.episodes; step++) {
                    Variable action = agent.selectAction(environment.getCurrentState());
                    Environment.StepResult result = environment.step(action);
                    
                    Experience experience = new Experience(
                        environment.getCurrentState(), action, result.getReward(),
                        result.getNextState(), result.isDone(), step
                    );
                    agent.learn(experience);
                    
                    // 计算瞬时悔恨
                    float instantRegret = maxReward - result.getReward();
                    cumulativeRegret += instantRegret;
                    
                    if (step % 100 == 99) {
                        cumulativeRegrets.add(cumulativeRegret);
                    }
                }
            }
            
            // 显示悔恨增长
            if (!cumulativeRegrets.isEmpty()) {
                System.out.printf("最终累积悔恨: %.4f±%.4f%n",
                                 cumulativeRegrets.stream().mapToDouble(Double::valueOf).average().orElse(0),
                                 calculateStandardDeviation(cumulativeRegrets));
            }
        }
    }
    
    /**
     * 学习曲线分析
     */
    private void performLearningCurveAnalysis(List<PerformanceMetrics> results) {
        System.out.println("\n=== 学习曲线分析 ===");
        
        for (PerformanceMetrics metrics : results) {
            System.out.println("\n算法: " + metrics.algorithmName);
            
            // 分析不同阶段的表现
            int episodeCount = metrics.episodeRewards.size() / 10; // 假设有10次运行
            if (episodeCount < 10) continue;
            
            int[] checkpoints = {episodeCount / 10, episodeCount / 4, episodeCount / 2, episodeCount};
            String[] labels = {"早期(10%)", "前期(25%)", "中期(50%)", "后期(100%)"};
            
            for (int i = 0; i < checkpoints.length; i++) {
                int endIndex = Math.min(checkpoints[i] * 10, metrics.episodeRewards.size());
                List<Float> subset = metrics.episodeRewards.subList(0, endIndex);
                double avgReward = subset.stream().mapToDouble(Double::valueOf).average().orElse(0);
                System.out.printf("  %s: 平均奖励 = %.4f%n", labels[i], avgReward);
            }
            
            // 学习速度分析
            analyzeLearningSpeed(metrics);
        }
    }
    
    /**
     * 分析学习速度
     */
    private void analyzeLearningSpeed(PerformanceMetrics metrics) {
        if (metrics.episodeRewards.size() < 100) return;
        
        // 计算改进速度（前10%与后10%的差异）
        int segmentSize = metrics.episodeRewards.size() / 10;
        
        List<Float> earlyRewards = metrics.episodeRewards.subList(0, segmentSize);
        List<Float> lateRewards = metrics.episodeRewards.subList(
            metrics.episodeRewards.size() - segmentSize, metrics.episodeRewards.size());
        
        double earlyAvg = earlyRewards.stream().mapToDouble(Double::valueOf).average().orElse(0);
        double lateAvg = lateRewards.stream().mapToDouble(Double::valueOf).average().orElse(0);
        double improvement = lateAvg - earlyAvg;
        
        System.out.printf("  学习改进: %.4f (从 %.4f 到 %.4f)%n", improvement, earlyAvg, lateAvg);
        
        if (improvement > 0) {
            System.out.printf("  改进幅度: %.2f%%%n", (improvement / Math.abs(earlyAvg)) * 100);
        }
    }
    
    /**
     * 生成综合报告
     */
    public void generateComprehensiveReport() {
        System.out.println("\n=======================================");
        System.out.println("           综合性能分析报告             ");
        System.out.println("=======================================");
        
        System.out.println("\n=== 算法选择建议 ===");
        System.out.println("1. 多臂老虎机问题:");
        System.out.println("   - 低探索需求: 推荐汤普森采样");
        System.out.println("   - 稳定性优先: 推荐UCB");
        System.out.println("   - 简单实现: 推荐ε-贪心(ε=0.05)");
        
        System.out.println("\n2. 连续控制问题(CartPole):");
        System.out.println("   - 样本效率: 推荐DQN");
        System.out.println("   - 策略梯度: 推荐REINFORCE+基线");
        
        System.out.println("\n3. 离散导航问题(GridWorld):");
        System.out.println("   - 小规模: 推荐DQN");
        System.out.println("   - 大规模: 考虑更高级的方法");
        
        printPerformanceSummary();
        printRecommendations();
    }
    
    /**
     * 打印性能摘要
     */
    private void printPerformanceSummary() {
        System.out.println("\n=== 性能特点摘要 ===");
        System.out.println("ε-贪心: 实现简单，参数敏感，适合快速原型");
        System.out.println("UCB: 理论保证好，自适应探索，适合在线学习");
        System.out.println("汤普森采样: 贝叶斯方法，探索效率高，适合随机环境");
        System.out.println("DQN: 深度学习，适合高维状态，需要经验回放");
        System.out.println("REINFORCE: 策略梯度，直接优化策略，适合连续动作");
    }
    
    /**
     * 打印使用建议
     */
    private void printRecommendations() {
        System.out.println("\n=== 使用建议 ===");
        System.out.println("1. 参数调优：");
        System.out.println("   - 学习率: 从0.01开始，根据收敛情况调整");
        System.out.println("   - 探索率: 从0.1开始，随训练衰减");
        System.out.println("   - 网络结构: 从简单开始，逐步增加复杂度");
        
        System.out.println("\n2. 训练监控：");
        System.out.println("   - 监控学习曲线，注意过拟合");
        System.out.println("   - 记录收敛点，评估训练效率");
        System.out.println("   - 进行多次独立运行，确保结果可靠");
        
        System.out.println("\n3. 实验设计：");
        System.out.println("   - 设置适当的对照组");
        System.out.println("   - 使用统计检验验证显著性");
        System.out.println("   - 记录实验细节，确保可重现性");
    }
    
    /**
     * 计算标准差
     */
    private double calculateStandardDeviation(List<Float> values) {
        double mean = values.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
        double variance = values.stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average().orElse(0.0);
        return Math.sqrt(variance);
    }
    
    /**
     * 智能体工厂接口
     */
    @FunctionalInterface
    private interface AgentFactory {
        Agent create();
    }
}