package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;
import io.leavesfly.tinyai.rl.Environment;

import java.util.*;

/**
 * 多臂老虎机算法对比演示
 *
 * 本演示对比三种经典多臂老虎机算法:
 * - ε-贪心: 固定概率探索
 * - UCB: 置信区间探索
 * - 汤普森采样: 贝叶斯采样
 *
 * 运行方式:
 *   mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.BanditAlgorithmsDemo" -pl tinyai-deeplearning-rl
 */
public class BanditAlgorithmsDemo {

    private static final int NUM_ARMS = 5;
    private static final int NUM_STEPS = 500;
    private static final int NUM_RUNS = 5;
    private static final int OPTIMAL_ARM_INDEX = 2;  // 对应 trueRewards 中 0.8 的老虎机

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("      多臂老虎机算法对比演示              ");
        System.out.println("==========================================\n");

        // 设置环境
        float[] trueRewards = {0.2f, 0.5f, 0.8f, 0.3f, 0.6f};
        
        System.out.println("【实验设置】");
        System.out.println("老虎机数量: " + NUM_ARMS);
        System.out.print("真实奖励期望: [");
        for (int i = 0; i < trueRewards.length; i++) {
            System.out.printf("%.1f%s", trueRewards[i], i < trueRewards.length - 1 ? ", " : "");
        }
        System.out.println("]");
        System.out.println("最优选择: 老虎机2 (奖励=0.8)");
        System.out.println("实验步数: " + NUM_STEPS);
        System.out.println("独立运行: " + NUM_RUNS + "次\n");

        // 测试三种算法
        Map<String, AlgorithmResults> results = new LinkedHashMap<>();
        
        System.out.println("【算法1: ε-贪心 (ε=0.1)】");
        results.put("ε-贪心", testEpsilonGreedy(trueRewards));
        
        System.out.println("\n【算法2: UCB】");
        results.put("UCB", testUCB(trueRewards));
        
        System.out.println("\n【算法3: 汤普森采样】");
        results.put("汤普森采样", testThompsonSampling(trueRewards));

        // 显示对比结果
        displayComparison(results, trueRewards);
    }

    /**
     * 测试ε-贪心算法
     */
    private static AlgorithmResults testEpsilonGreedy(float[] trueRewards) {
        System.out.println("原理: 以概率ε=0.1随机探索,以概率0.9选择当前最优");
        
        AlgorithmResults results = new AlgorithmResults();
        
        for (int run = 0; run < NUM_RUNS; run++) {
            MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(trueRewards, NUM_STEPS);
            EpsilonGreedyBanditAgent agent = new EpsilonGreedyBanditAgent("ε-Greedy", NUM_ARMS, 0.1f);
            
            RunResult runResult = runExperiment(env, agent);
            results.addRun(runResult);
            
            if (run == 0) {
                System.out.println("第1次运行详情:");
                printRunSummary(runResult);
            }
        }
        results.computeStatistics();
        return results;
    }

    /**
     * 测试UCB算法
     */
    private static AlgorithmResults testUCB(float[] trueRewards) {
        System.out.println("原理: 基于上置信区间选择,考虑均值估计和不确定性");
        
        AlgorithmResults results = new AlgorithmResults();
        
        for (int run = 0; run < NUM_RUNS; run++) {
            MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(trueRewards, NUM_STEPS);
            UCBBanditAgent agent = new UCBBanditAgent("UCB", NUM_ARMS);
            
            RunResult runResult = runExperiment(env, agent);
            results.addRun(runResult);
            
            if (run == 0) {
                System.out.println("第1次运行详情:");
                printRunSummary(runResult);
            }
        }
        results.computeStatistics();
        return results;
    }

    /**
     * 测试汤普森采样算法
     */
    private static AlgorithmResults testThompsonSampling(float[] trueRewards) {
        System.out.println("原理: 贝叶斯方法,从后验分布采样进行决策");
        
        AlgorithmResults results = new AlgorithmResults();
        
        for (int run = 0; run < NUM_RUNS; run++) {
            MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(trueRewards, NUM_STEPS);
            ThompsonSamplingBanditAgent agent = new ThompsonSamplingBanditAgent("Thompson", NUM_ARMS);
            
            RunResult runResult = runExperiment(env, agent);
            results.addRun(runResult);
            
            if (run == 0) {
                System.out.println("第1次运行详情:");
                printRunSummary(runResult);
            }
        }
        results.computeStatistics();
        return results;
    }

    /**
     * 运行单次实验
     */
    private static RunResult runExperiment(Environment env, BanditAgent agent) {
        Variable state = env.reset();
        float totalReward = 0;
        int optimalActions = 0;
        int[] actionCounts = new int[NUM_ARMS];
        
        for (int step = 0; step < NUM_STEPS; step++) {
            Variable action = agent.selectAction(state);
            int selectedArm = (int) action.getValue().getNumber().floatValue();
            actionCounts[selectedArm]++;
            
            Environment.StepResult result = env.step(action);
            float reward = result.getReward();
            totalReward += reward;
            
            if (selectedArm == OPTIMAL_ARM_INDEX) {
                optimalActions++;
            }
            
            Experience experience = new Experience(
                state, action, reward,
                result.getNextState(), result.isDone(), step
            );
            agent.learn(experience);
            
            state = result.getNextState();
        }
        
        return new RunResult(totalReward, optimalActions, actionCounts);
    }

    /**
     * 打印单次运行摘要
     */
    private static void printRunSummary(RunResult result) {
        System.out.printf("  总奖励: %.2f, 平均奖励: %.4f\n", 
            result.totalReward, result.totalReward / NUM_STEPS);
        System.out.printf("  最优选择率: %.2f%% (%d/%d)\n",
            (float) result.optimalActions / NUM_STEPS * 100,
            result.optimalActions, NUM_STEPS);
        
        System.out.print("  选择分布: [");
        for (int i = 0; i < result.actionCounts.length; i++) {
            System.out.printf("%d%s", result.actionCounts[i], 
                i < result.actionCounts.length - 1 ? ", " : "");
        }
        System.out.println("]");
    }

    /**
     * 显示算法对比结果
     */
    private static void displayComparison(Map<String, AlgorithmResults> results, float[] trueRewards) {
        System.out.println("\n==========================================");
        System.out.println("           对比结果分析                   ");
        System.out.println("==========================================\n");

        System.out.println("【性能指标对比】");
        System.out.printf("%-12s | %12s | %12s | %15s\n", 
            "算法", "平均奖励", "标准差", "最优选择率");
        System.out.println("-------------|-------------|-------------|---------------");
        
        for (Map.Entry<String, AlgorithmResults> entry : results.entrySet()) {
            String name = entry.getKey();
            AlgorithmResults res = entry.getValue();
            
            System.out.printf("%-12s | %12.4f | %12.4f | %14.2f%%\n",
                name, res.meanReward, res.stdReward, res.meanOptimalRate * 100);
        }

        // 悔恨分析
        System.out.println("\n【悔恨(Regret)分析】");
        float optimalReward = 0;
        for (float reward : trueRewards) {
            if (reward > optimalReward) {
                optimalReward = reward;
            }
        }
        System.out.println("理论最优奖励: " + optimalReward);
        System.out.println();
        
        for (Map.Entry<String, AlgorithmResults> entry : results.entrySet()) {
            String name = entry.getKey();
            AlgorithmResults res = entry.getValue();
            float totalOptimalReward = optimalReward * NUM_STEPS;
            float totalActualReward = (float) res.meanReward * NUM_STEPS;
            float regret = (float) (totalOptimalReward - totalActualReward);
            
            System.out.printf("%-12s: 累积悔恨 = %.2f\n", name, regret);
        }

        // 算法特点总结
        System.out.println("\n【算法特点总结】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("\nε-贪心:");
        System.out.println("  优点: 简单易懂,实现容易,计算开销小");
        System.out.println("  缺点: 探索是随机的,效率可能不高");
        System.out.println("  适用: 快速原型,在线学习,简单环境");
        
        System.out.println("\nUCB:");
        System.out.println("  优点: 理论保证好,探索更有针对性");
        System.out.println("  缺点: 对参数敏感,计算稍复杂");
        System.out.println("  适用: 需要理论保证,稳定环境");
        
        System.out.println("\n汤普森采样:");
        System.out.println("  优点: 贝叶斯方法,自然平衡探索与利用");
        System.out.println("  缺点: 需要假设奖励分布,计算相对复杂");
        System.out.println("  适用: 贝叶斯优化,不确定性建模");

        System.out.println("\n【推荐选择】");
        String best = results.entrySet().stream()
            .max(Comparator.comparingDouble(e -> e.getValue().meanReward))
            .map(Map.Entry::getKey)
            .orElse("未知");
        
        System.out.println("本次实验最佳算法: " + best);
        System.out.println("\n实际应用建议:");
        System.out.println("  • 追求简单: 选择ε-贪心");
        System.out.println("  • 追求理论: 选择UCB");
        System.out.println("  • 追求效果: 选择汤普森采样");
    }

    /**
     * 单次运行结果
     */
    private static class RunResult {
        float totalReward;
        int optimalActions;
        int[] actionCounts;
        
        RunResult(float totalReward, int optimalActions, int[] actionCounts) {
            this.totalReward = totalReward;
            this.optimalActions = optimalActions;
            this.actionCounts = actionCounts;
        }
    }

    /**
     * 算法总体结果
     */
    private static class AlgorithmResults {
        List<Float> rewards = new ArrayList<>();
        List<Float> optimalRates = new ArrayList<>();
        
        double meanReward;
        double stdReward;
        double meanOptimalRate;
        
        void addRun(RunResult result) {
            float avgReward = result.totalReward / NUM_STEPS;
            float optimalRate = (float) result.optimalActions / NUM_STEPS;
            
            rewards.add(avgReward);
            optimalRates.add(optimalRate);
        }
        
        void computeStatistics() {
            meanReward = rewards.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
            meanOptimalRate = optimalRates.stream().mapToDouble(Double::valueOf).average().orElse(0.0);
            
            double variance = rewards.stream()
                .mapToDouble(r -> Math.pow(r - meanReward, 2))
                .average().orElse(0.0);
            stdReward = Math.sqrt(variance);
        }
    }
    
}
