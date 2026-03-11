package io.leavesfly.tinyai.rl.demo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.*;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;
import io.leavesfly.tinyai.rl.Environment;

/**
 * 算法对比分析演示 - 并排比较不同强化学习算法
 *
 * 本演示对比:
 * - ε-贪心 vs UCB vs 汤普森采样
 * - 不同探索策略的学习曲线
 * - 累积遗憾(Cumulative Regret)对比
 * - 收敛速度分析
 *
 * 运行方式:
 *   mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.rl.demo.AlgorithmComparisonDemo" -pl tinyai-deeplearning-rl
 */
public class AlgorithmComparisonDemo {

    private static final int NUM_ARMS = 5;
    private static final int MAX_STEPS = 500;
    private static final int NUM_RUNS = 10; // 多次运行取平均

    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("      强化学习算法对比分析               ");
        System.out.println("==========================================\n");

        // 设置奖励分布
        float[] rewards = {0.1f, 0.3f, 0.5f, 0.7f, 0.4f};
        
        System.out.println("实验设置:");
        System.out.println("  老虎机数量: " + NUM_ARMS);
        System.out.println("  训练步数: " + MAX_STEPS);
        System.out.println("  运行次数: " + NUM_RUNS + " (取平均)");
        System.out.println("\n真实奖励分布:");
        for (int i = 0; i < rewards.length; i++) {
            System.out.printf("  老虎机 %c: %.1f%%%n", (char)('A' + i), rewards[i] * 100);
        }
        System.out.println("  (老虎机D是最优选择)");

        // 运行对比实验
        System.out.println("\n开始对比实验...\n");
        
        AlgorithmResult epsilonGreedy = runMultipleExperiments(
            "ε-贪心", rewards, () -> new EpsilonGreedyBanditAgent("ε-贪心", NUM_ARMS, 0.1f)
        );
        
        AlgorithmResult ucb = runMultipleExperiments(
            "UCB", rewards, () -> new UCBBanditAgent("UCB", NUM_ARMS, 2.0f)
        );
        
        AlgorithmResult thompson = runMultipleExperiments(
            "汤普森采样", rewards, () -> new ThompsonSamplingBanditAgent("汤普森", NUM_ARMS)
        );

        // 显示对比结果
        displayComparison(epsilonGreedy, ucb, thompson, rewards);
    }

    /**
     * 多次运行实验取平均
     */
    private static AlgorithmResult runMultipleExperiments(
            String name, float[] rewards, AgentFactory agentFactory) {
        
        float[] avgRewards = new float[MAX_STEPS];
        float[] avgRegrets = new float[MAX_STEPS];
        float[] avgOptimalRates = new float[MAX_STEPS];
        
        for (int run = 0; run < NUM_RUNS; run++) {
            SingleRunResult result = runSingleExperiment(rewards, agentFactory);
            
            for (int i = 0; i < MAX_STEPS; i++) {
                avgRewards[i] += result.rewards[i];
                avgRegrets[i] += result.regrets[i];
                avgOptimalRates[i] += result.optimalChoices[i];
            }
        }
        
        // 计算平均
        for (int i = 0; i < MAX_STEPS; i++) {
            avgRewards[i] /= NUM_RUNS;
            avgRegrets[i] /= NUM_RUNS;
            avgOptimalRates[i] /= NUM_RUNS;
        }
        
        return new AlgorithmResult(name, avgRewards, avgRegrets, avgOptimalRates);
    }

    /**
     * 运行单次实验
     */
    private static SingleRunResult runSingleExperiment(float[] rewards, AgentFactory agentFactory) {
        BanditAgent agent = agentFactory.create();
        MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(rewards, MAX_STEPS);
        
        float[] stepRewards = new float[MAX_STEPS];
        float[] stepRegrets = new float[MAX_STEPS];
        float[] optimalChoices = new float[MAX_STEPS];
        
        // 找出最优动作
        int optimalAction = 0;
        float maxReward = rewards[0];
        for (int i = 1; i < rewards.length; i++) {
            if (rewards[i] > maxReward) {
                maxReward = rewards[i];
                optimalAction = i;
            }
        }
        
        Variable state = env.reset();
        float cumulativeRegret = 0;
        int optimalCount = 0;
        
        for (int step = 0; step < MAX_STEPS; step++) {
            Variable action = agent.selectAction(state);
            int selectedArm = (int) action.getValue().getNumber().floatValue();
            
            Environment.StepResult result = env.step(action);
            float reward = result.getReward();
            
            // 计算遗憾
            float regret = maxReward - reward;
            cumulativeRegret += regret;
            
            // 统计最优选择
            if (selectedArm == optimalAction) {
                optimalCount++;
            }
            
            // 学习
            Experience experience = new Experience(
                state, action, reward, result.getNextState(), result.isDone(), step
            );
            agent.learn(experience);
            
            stepRewards[step] = reward;
            stepRegrets[step] = cumulativeRegret;
            optimalChoices[step] = (float) optimalCount / (step + 1);
            
            state = result.getNextState();
        }
        
        return new SingleRunResult(stepRewards, stepRegrets, optimalChoices);
    }

    /**
     * 显示对比结果
     */
    private static void displayComparison(AlgorithmResult epsilonGreedy, 
                                         AlgorithmResult ucb, 
                                         AlgorithmResult thompson,
                                         float[] trueRewards) {
        
        System.out.println("\n【对比结果】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 1. 最终性能对比
        System.out.println("\n1. 最终性能对比:");
        System.out.println("────────────────────────────────────────");
        System.out.printf("%-15s | %-12s | %-12s | %-12s%n", 
            "算法", "累积遗憾", "最优选择率", "平均奖励");
        System.out.println("────────────────────────────────────────");
        
        printAlgorithmSummary("ε-贪心", epsilonGreedy, trueRewards);
        printAlgorithmSummary("UCB", ucb, trueRewards);
        printAlgorithmSummary("汤普森采样", thompson, trueRewards);
        
        // 2. 学习曲线对比
        System.out.println("\n2. 累积遗憾曲线 (关键节点):");
        System.out.println("────────────────────────────────────────");
        System.out.printf("%-10s | %-10s | %-10s | %-10s%n", 
            "步数", "ε-贪心", "UCB", "汤普森采样");
        System.out.println("────────────────────────────────────────");
        
        int[] checkpoints = {50, 100, 200, 300, 400, 500};
        for (int step : checkpoints) {
            if (step <= MAX_STEPS) {
                System.out.printf("%-10d | %-10.1f | %-10.1f | %-10.1f%n",
                    step,
                    epsilonGreedy.regrets[step - 1],
                    ucb.regrets[step - 1],
                    thompson.regrets[step - 1]);
            }
        }
        
        // 3. 最优选择率曲线
        System.out.println("\n3. 最优选择率曲线 (关键节点):");
        System.out.println("────────────────────────────────────────");
        System.out.printf("%-10s | %-10s | %-10s | %-10s%n", 
            "步数", "ε-贪心", "UCB", "汤普森采样");
        System.out.println("────────────────────────────────────────");
        
        for (int step : checkpoints) {
            if (step <= MAX_STEPS) {
                System.out.printf("%-10d | %-9.1f%% | %-9.1f%% | %-9.1f%%%n",
                    step,
                    epsilonGreedy.optimalRates[step - 1] * 100,
                    ucb.optimalRates[step - 1] * 100,
                    thompson.optimalRates[step - 1] * 100);
            }
        }
        
        // 4. 可视化曲线
        System.out.println("\n4. 累积遗憾可视化:");
        System.out.println("────────────────────────────────────────");
        visualizeCurves(epsilonGreedy.regrets, ucb.regrets, thompson.regrets, "遗憾");
        
        System.out.println("\n5. 最优选择率可视化:");
        System.out.println("────────────────────────────────────────");
        visualizeOptimalRates(epsilonGreedy.optimalRates, ucb.optimalRates, thompson.optimalRates);
        
        // 5. 算法分析
        System.out.println("\n【算法分析】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        System.out.println("\nε-贪心算法:");
        System.out.println("  优点: 简单直观，实现容易");
        System.out.println("  缺点: 探索率固定，后期仍可能随机选择");
        System.out.println("  适用: 简单环境，快速原型");
        
        System.out.println("\nUCB算法:");
        System.out.println("  优点: 理论保证，遗憾界最优");
        System.out.println("  缺点: 需要调节参数c，对非平稳环境敏感");
        System.out.println("  适用: 稳定环境，需要理论保证");
        
        System.out.println("\n汤普森采样:");
        System.out.println("  优点: 贝叶斯方法，自然处理不确定性");
        System.out.println("  缺点: 计算复杂度较高");
        System.out.println("  适用: 复杂环境，需要不确定性建模");
        
        // 6. 推荐
        System.out.println("\n【推荐建议】");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // 找出最佳算法
        float epsilonFinalRegret = epsilonGreedy.regrets[MAX_STEPS - 1];
        float ucbFinalRegret = ucb.regrets[MAX_STEPS - 1];
        float thompsonFinalRegret = thompson.regrets[MAX_STEPS - 1];
        
        String bestAlgorithm;
        float minRegret = Math.min(epsilonFinalRegret, Math.min(ucbFinalRegret, thompsonFinalRegret));
        if (minRegret == epsilonFinalRegret) {
            bestAlgorithm = "ε-贪心";
        } else if (minRegret == ucbFinalRegret) {
            bestAlgorithm = "UCB";
        } else {
            bestAlgorithm = "汤普森采样";
        }
        
        System.out.printf("根据本次实验，%s 表现最佳 (累积遗憾: %.1f)%n", 
            bestAlgorithm, minRegret);
        System.out.println("\n实际应用建议:");
        System.out.println("  • 快速实现: 选择 ε-贪心");
        System.out.println("  • 理论保证: 选择 UCB");
        System.out.println("  • 复杂场景: 选择 汤普森采样");
    }

    /**
     * 打印算法摘要
     */
    private static void printAlgorithmSummary(String name, AlgorithmResult result, float[] trueRewards) {
        float finalRegret = result.regrets[MAX_STEPS - 1];
        float finalOptimalRate = result.optimalRates[MAX_STEPS - 1];
        float avgReward = 0;
        for (float r : result.rewards) {
            avgReward += r;
        }
        avgReward /= result.rewards.length;
        
        System.out.printf("%-15s | %-12.1f | %-11.1f%% | %-12.3f%n",
            name, finalRegret, finalOptimalRate * 100, avgReward);
    }

    /**
     * 可视化曲线
     */
    private static void visualizeCurves(float[] epsilon, float[] ucb, float[] thompson, String label) {
        float maxValue = Math.max(epsilon[MAX_STEPS - 1], 
                         Math.max(ucb[MAX_STEPS - 1], thompson[MAX_STEPS - 1]));
        
        int[] steps = {0, 100, 200, 300, 400, 499};
        String[] labels = {"0", "100", "200", "300", "400", "500"};
        
        // 绘制坐标轴
        System.out.println("        " + String.join("    ", labels));
        System.out.println("        " + "─".repeat(30));
        
        String[] names = {"ε-贪心", "UCB  ", "汤普森"};
        float[][] data = {epsilon, ucb, thompson};
        
        for (int i = 0; i < 3; i++) {
            System.out.printf("%s |", names[i]);
            for (int step : steps) {
                int height = (int) (data[i][step] / maxValue * 5);
                char c = " ▁▂▃▄▅".charAt(Math.min(height, 5));
                System.out.print(c + "    ");
            }
            System.out.printf(" (%.0f)%n", data[i][MAX_STEPS - 1]);
        }
    }

    /**
     * 可视化最优选择率
     */
    private static void visualizeOptimalRates(float[] epsilon, float[] ucb, float[] thompson) {
        int[] steps = {0, 100, 200, 300, 400, 499};
        String[] labels = {"0", "100", "200", "300", "400", "500"};
        
        System.out.println("        " + String.join("    ", labels));
        System.out.println("        " + "─".repeat(30));
        
        String[] names = {"ε-贪心", "UCB  ", "汤普森"};
        float[][] data = {epsilon, ucb, thompson};
        
        for (int i = 0; i < 3; i++) {
            System.out.printf("%s |", names[i]);
            for (int step : steps) {
                int height = (int) (data[i][step] * 5);
                char c = " ▁▂▃▄▅".charAt(Math.min(height, 5));
                System.out.print(c + "    ");
            }
            System.out.printf(" (%.0f%%)%n", data[i][MAX_STEPS - 1] * 100);
        }
    }

    /**
     * 智能体工厂接口
     */
    @FunctionalInterface
    private interface AgentFactory {
        BanditAgent create();
    }

    /**
     * 单次运行结果
     */
    private static class SingleRunResult {
        final float[] rewards;
        final float[] regrets;
        final float[] optimalChoices;
        
        SingleRunResult(float[] rewards, float[] regrets, float[] optimalChoices) {
            this.rewards = rewards;
            this.regrets = regrets;
            this.optimalChoices = optimalChoices;
        }
    }

    /**
     * 算法对比结果
     */
    private static class AlgorithmResult {
        final String name;
        final float[] rewards;
        final float[] regrets;
        final float[] optimalRates;
        
        AlgorithmResult(String name, float[] rewards, float[] regrets, float[] optimalRates) {
            this.name = name;
            this.rewards = rewards;
            this.regrets = regrets;
            this.optimalRates = optimalRates;
        }
    }
}
