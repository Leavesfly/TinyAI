package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.BanditAgent;
import io.leavesfly.tinyai.rl.agent.EpsilonGreedyBanditAgent;
import io.leavesfly.tinyai.rl.agent.ThompsonSamplingBanditAgent;
import io.leavesfly.tinyai.rl.agent.UCBBanditAgent;
import io.leavesfly.tinyai.rl.environment.MultiArmedBanditEnvironment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 多臂老虎机算法比较示例
 * <p>
 * 本示例比较三种经典的多臂老虎机算法：
 * 1. ε-贪心算法 (Epsilon-Greedy)
 * 2. UCB算法 (Upper Confidence Bound)
 * 3. 汤普森采样 (Thompson Sampling)
 * <p>
 * 评估指标：
 * - 累积奖励 (Cumulative Reward)
 * - 累积悔恨 (Cumulative Regret)
 * - 最优动作选择率 (Optimal Action Rate)
 *
 * @author leavesfly
 */
public class MultiArmedBanditExample {

    /**
     * 实验参数
     */
    private static final int NUM_ARMS = 5;           // 臂的数量
    private static final int NUM_STEPS = 1000;      // 实验步数
    private static final int NUM_RUNS = 10;         // 独立运行次数

    /**
     * 真实奖励设置（每个臂的期望奖励）
     */
    private static final float[] TRUE_REWARDS = {0.2f, 0.5f, 0.8f, 0.3f, 0.6f};

    public static void main(String[] args) {
        System.out.println("======== 多臂老虎机算法比较实验 ========");
        System.out.println("实验设置:");
        System.out.println("  臂数量: " + NUM_ARMS);
        System.out.println("  实验步数: " + NUM_STEPS);
        System.out.println("  独立运行次数: " + NUM_RUNS);
        System.out.print("  真实奖励: [");
        for (int i = 0; i < TRUE_REWARDS.length; i++) {
            System.out.print(String.format("%.2f", TRUE_REWARDS[i]));
            if (i < TRUE_REWARDS.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        System.out.println("  最优臂: " + getOptimalArm() + " (奖励: " + String.format("%.2f", getOptimalReward()) + ")");
        System.out.println();

        // 运行比较实验
        runComparisonExperiment();

        // 运行单次详细实验
        System.out.println("\n======== 单次详细实验 ========");
        runDetailedSingleExperiment();
    }

    /**
     * 运行算法比较实验
     */
    private static void runComparisonExperiment() {
        System.out.println("开始多次运行实验...\n");

        // 存储每次运行的结果
        Map<String, List<ExperimentResult>> allResults = new HashMap<>();

        for (int run = 0; run < NUM_RUNS; run++) {
            System.out.println("运行 " + (run + 1) + "/" + NUM_RUNS);

            // 创建智能体
            List<BanditAgent> agents = createAgents();

            for (BanditAgent agent : agents) {
                ExperimentResult result = runSingleExperiment(agent, run);
                allResults.computeIfAbsent(agent.getName(), k -> new ArrayList<>()).add(result);
            }
        }

        // 计算并显示平均结果
        System.out.println("\n======== 实验结果总结 ========");
        displayAverageResults(allResults);
    }

    /**
     * 运行单次详细实验
     */
    private static void runDetailedSingleExperiment() {
        List<BanditAgent> agents = createAgents();

        for (BanditAgent agent : agents) {
            System.out.println("\n--- " + agent.getName() + " 详细过程 ---");
            runDetailedExperiment(agent);
        }
    }

    /**
     * 创建要比较的智能体列表
     */
    private static List<BanditAgent> createAgents() {
        List<BanditAgent> agents = new ArrayList<>();

        // ε-贪心算法（三种不同的ε值）
        agents.add(new EpsilonGreedyBanditAgent("ε-贪心(ε=0.1)", NUM_ARMS, 0.1f));
        agents.add(new EpsilonGreedyBanditAgent("ε-贪心(ε=0.05)", NUM_ARMS, 0.05f));

        // UCB算法
        agents.add(new UCBBanditAgent("UCB", NUM_ARMS));

        // 汤普森采样
        agents.add(new ThompsonSamplingBanditAgent("汤普森采样", NUM_ARMS));

        return agents;
    }

    /**
     * 运行单次实验
     */
    private static ExperimentResult runSingleExperiment(BanditAgent agent, int runIndex) {
        // 重置智能体
        agent.reset();

        // 为支持随机种子的智能体设置种子
        if (agent instanceof EpsilonGreedyBanditAgent) {
            ((EpsilonGreedyBanditAgent) agent).setSeed(runIndex * 100L);
        } else if (agent instanceof ThompsonSamplingBanditAgent) {
            ((ThompsonSamplingBanditAgent) agent).setSeed(runIndex * 100L);
        }

        // 创建环境
        MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(TRUE_REWARDS, NUM_STEPS);
        env.setSeed(runIndex * 100L);

        // 初始化环境
        env.reset();

        float totalReward = 0.0f;
        float totalRegret = 0.0f;
        int optimalActions = 0;
        int optimalArm = getOptimalArm();

        // 运行实验
        for (int step = 0; step < NUM_STEPS; step++) {
            // 智能体选择动作
            Variable action = agent.selectAction(env.getCurrentState());

            // 环境执行动作
            Environment.StepResult result = env.step(action);

            // 创建经验并让智能体学习
            Experience experience = new Experience(
                    env.getCurrentState(),
                    action,
                    result.getReward(),
                    result.getNextState(),
                    result.isDone(),
                    step
            );
            agent.learn(experience);

            // 统计指标
            totalReward += result.getReward();
            totalRegret += (float) result.getInfo().get("instantRegret");

            int selectedArm = (int) result.getInfo().get("selectedArm");
            if (selectedArm == optimalArm) {
                optimalActions++;
            }
        }

        float optimalActionRate = (float) optimalActions / NUM_STEPS;

        return new ExperimentResult(agent.getName(), totalReward, totalRegret, optimalActionRate);
    }

    /**
     * 运行详细实验（显示中间过程）
     */
    private static void runDetailedExperiment(BanditAgent agent) {
        agent.reset();

        MultiArmedBanditEnvironment env = new MultiArmedBanditEnvironment(TRUE_REWARDS, 100);
        env.reset();

        // 显示前50步的详细过程
        for (int step = 0; step < 50; step++) {
            Variable action = agent.selectAction(env.getCurrentState());
            Environment.StepResult result = env.step(action);

            Experience experience = new Experience(
                    env.getCurrentState(),
                    action,
                    result.getReward(),
                    result.getNextState(),
                    result.isDone(),
                    step
            );
            agent.learn(experience);

            if (step % 10 == 0 || step < 10) {
                int selectedArm = (int) result.getInfo().get("selectedArm");
                System.out.println(String.format("步骤 %3d: 选择臂 %d, 奖励 %.4f, 悔恨 %.4f",
                        step + 1, selectedArm, result.getReward(),
                        (float) result.getInfo().get("instantRegret")));
            }
        }

        // 显示最终状态
        System.out.println("\n最终状态:");
        agent.printStatus();
    }

    /**
     * 显示平均结果
     */
    private static void displayAverageResults(Map<String, List<ExperimentResult>> allResults) {
        System.out.printf("%-20s | %12s | %12s | %15s%n",
                "算法", "平均累积奖励", "平均累积悔恨", "平均最优选择率");
        System.out.println("-----|-------------|-------------|---------------");

        List<String> sortedAgents = new ArrayList<>(allResults.keySet());
        sortedAgents.sort((a, b) -> {
            double avgRewardA = allResults.get(a).stream().mapToDouble(ExperimentResult::getTotalReward).average().orElse(0);
            double avgRewardB = allResults.get(b).stream().mapToDouble(ExperimentResult::getTotalReward).average().orElse(0);
            return Double.compare(avgRewardB, avgRewardA); // 按奖励降序排列
        });

        for (String agentName : sortedAgents) {
            List<ExperimentResult> results = allResults.get(agentName);

            double avgReward = results.stream().mapToDouble(ExperimentResult::getTotalReward).average().orElse(0);
            double avgRegret = results.stream().mapToDouble(ExperimentResult::getTotalRegret).average().orElse(0);
            double avgOptimalRate = results.stream().mapToDouble(ExperimentResult::getOptimalActionRate).average().orElse(0);

            System.out.printf("%-20s | %12.4f | %12.4f | %14.2f%%%n",
                    agentName, avgReward, avgRegret, avgOptimalRate * 100);
        }

        System.out.println("\n算法排名分析:");
        System.out.println("1. 累积奖励越高越好");
        System.out.println("2. 累积悔恨越低越好");
        System.out.println("3. 最优选择率越高越好");
    }

    /**
     * 获取最优臂的索引
     */
    private static int getOptimalArm() {
        int optimalArm = 0;
        for (int i = 1; i < TRUE_REWARDS.length; i++) {
            if (TRUE_REWARDS[i] > TRUE_REWARDS[optimalArm]) {
                optimalArm = i;
            }
        }
        return optimalArm;
    }

    /**
     * 获取最优奖励
     */
    private static float getOptimalReward() {
        return TRUE_REWARDS[getOptimalArm()];
    }

    /**
     * 实验结果数据类
     */
    private static class ExperimentResult {
        private final String agentName;
        private final float totalReward;
        private final float totalRegret;
        private final float optimalActionRate;

        public ExperimentResult(String agentName, float totalReward, float totalRegret, float optimalActionRate) {
            this.agentName = agentName;
            this.totalReward = totalReward;
            this.totalRegret = totalRegret;
            this.optimalActionRate = optimalActionRate;
        }

        public String getAgentName() {
            return agentName;
        }

        public float getTotalReward() {
            return totalReward;
        }

        public float getTotalRegret() {
            return totalRegret;
        }

        public float getOptimalActionRate() {
            return optimalActionRate;
        }
    }
}