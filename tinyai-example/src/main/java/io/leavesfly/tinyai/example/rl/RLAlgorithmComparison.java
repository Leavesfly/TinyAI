package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DQNAgent;
import io.leavesfly.tinyai.rl.agent.REINFORCEAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;
import io.leavesfly.tinyai.rl.environment.GridWorldEnvironment;

import java.util.ArrayList;
import java.util.List;

/**
 * 强化学习算法比较示例
 *
 * @author leavesfly
 * @version 0.01
 * <p>
 * 这个示例比较了不同强化学习算法在相同环境下的性能表现。
 * 目的是帮助理解不同算法的特点和适用场景。
 * <p>
 * 比较内容：
 * 1. DQN vs REINFORCE 在CartPole环境下的表现
 * 2. 不同超参数对算法性能的影响
 * 3. 训练速度和最终性能的权衡
 * 4. 算法在不同环境下的适应性
 */
public class RLAlgorithmComparison {

    public static void main(String[] args) {
        System.out.println("=== 强化学习算法比较实验 ===");

        // 实验1：CartPole环境下的算法比较
        System.out.println("\n【实验1】CartPole环境 - DQN vs REINFORCE");
        compareAlgorithmsOnCartPole();

        // 实验2：GridWorld环境下的算法比较
        System.out.println("\n【实验2】GridWorld环境 - 算法适应性分析");
        compareAlgorithmsOnGridWorld();

        // 实验3：超参数敏感性分析
        System.out.println("\n【实验3】超参数敏感性分析");
        hyperparameterSensitivityAnalysis();

        System.out.println("\n=== 比较实验完成 ===");
    }

    /**
     * 在CartPole环境下比较DQN和REINFORCE
     */
    private static void compareAlgorithmsOnCartPole() {
        Environment env = new CartPoleEnvironment(42L);
        int numEpisodes = 500;
        int numTrials = 3; // 多次试验求平均

        System.out.println("环境：CartPole");
        System.out.println("训练回合：" + numEpisodes);
        System.out.println("试验次数：" + numTrials);

        // DQN结果
        System.out.println("\n--- DQN 算法 ---");
        List<Float> dqnResults = new ArrayList<>();
        for (int trial = 0; trial < numTrials; trial++) {
            System.out.println("试验 " + (trial + 1) + ":");
            DQNAgent dqnAgent = new DQNAgent(
                    "DQN_Trial" + trial,
                    env.getStateDim(), env.getActionDim(),
                    new int[]{128, 128},
                    0.001f, 1.0f, 0.99f,
                    32, 10000, 100
            );
            float avgReward = trainAndEvaluate(dqnAgent, env, numEpisodes, false);
            dqnResults.add(avgReward);
            System.out.printf("  最终平均奖励: %.2f\n", avgReward);
        }

        // REINFORCE结果
        System.out.println("\n--- REINFORCE 算法 ---");
        List<Float> reinforceResults = new ArrayList<>();
        for (int trial = 0; trial < numTrials; trial++) {
            System.out.println("试验 " + (trial + 1) + ":");
            REINFORCEAgent reinforceAgent = new REINFORCEAgent(
                    "REINFORCE_Trial" + trial,
                    env.getStateDim(), env.getActionDim(),
                    new int[]{128, 128},
                    0.01f, 0.99f, true
            );
            float avgReward = trainAndEvaluate(reinforceAgent, env, numEpisodes, true);
            reinforceResults.add(avgReward);
            System.out.printf("  最终平均奖励: %.2f\n", avgReward);
        }

        // 统计分析
        printStatistics("DQN", dqnResults);
        printStatistics("REINFORCE", reinforceResults);
    }

    /**
     * 在GridWorld环境下比较算法适应性
     */
    private static void compareAlgorithmsOnGridWorld() {
        // 创建不同复杂度的GridWorld环境
        Environment simpleEnv = GridWorldEnvironment.createSimpleMaze();
        Environment complexEnv = GridWorldEnvironment.createWithRandomObstacles(6, 6, 0.3f);

        System.out.println("测试算法在不同复杂度环境下的适应性...");

        // 简单环境
        System.out.println("\n--- 简单迷宫环境 ---");
        testAlgorithmAdaptability(simpleEnv, "简单迷宫");

        // 复杂环境
        System.out.println("\n--- 复杂迷宫环境 ---");
        testAlgorithmAdaptability(complexEnv, "复杂迷宫");
    }

    /**
     * 测试算法在特定环境下的适应性
     *
     * @param env     环境
     * @param envName 环境名称
     */
    private static void testAlgorithmAdaptability(Environment env, String envName) {
        int numEpisodes = 1000;

        System.out.println("环境：" + envName);
        env.reset();
        env.render();

        // 测试REINFORCE（更适合离散动作空间）
        System.out.println("REINFORCE算法测试...");
        REINFORCEAgent reinforceAgent = new REINFORCEAgent(
                "REINFORCE_" + envName,
                env.getStateDim(), env.getActionDim(),
                new int[]{64, 64},
                0.01f, 0.99f, true
        );

        float reinforceScore = trainAndEvaluate(reinforceAgent, env, numEpisodes, true);
        System.out.printf("REINFORCE 最终得分: %.3f\n", reinforceScore);
    }

    /**
     * 超参数敏感性分析
     */
    private static void hyperparameterSensitivityAnalysis() {
        Environment env = new CartPoleEnvironment(123L);
        int numEpisodes = 300;

        System.out.println("分析DQN算法对不同学习率的敏感性...");

        float[] learningRates = {0.0001f, 0.001f, 0.01f, 0.1f};

        for (float lr : learningRates) {
            System.out.printf("\n学习率: %.4f\n", lr);

            DQNAgent agent = new DQNAgent(
                    "DQN_LR_" + lr,
                    env.getStateDim(), env.getActionDim(),
                    new int[]{64, 64},
                    lr, 1.0f, 0.99f,
                    32, 5000, 50
            );

            float result = trainAndEvaluate(agent, env, numEpisodes, false);
            System.out.printf("结果: %.2f\n", result);
        }

        System.out.println("\n分析完成！建议学习率在0.001左右。");
    }

    /**
     * 训练并评估智能体
     *
     * @param agent       智能体（可以是DQN或REINFORCE）
     * @param env         环境
     * @param numEpisodes 训练回合数
     * @param isReinforce 是否为REINFORCE算法
     * @return 平均评估奖励
     */
    private static float trainAndEvaluate(Object agent, Environment env, int numEpisodes, boolean isReinforce) {
        // 训练阶段
        for (int episode = 0; episode < numEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;

            while (!env.isDone()) {
                Variable action;
                if (isReinforce) {
                    action = ((REINFORCEAgent) agent).selectAction(state);
                } else {
                    action = ((DQNAgent) agent).selectAction(state);
                }

                Environment.StepResult result = env.step(action);
                Variable nextState = result.getNextState();
                float reward = result.getReward();
                boolean done = result.isDone();

                Experience experience = new Experience(state, action, reward, nextState, done);

                if (isReinforce) {
                    ((REINFORCEAgent) agent).learn(experience);
                } else {
                    ((DQNAgent) agent).learn(experience);
                }

                state = nextState;
                episodeReward += reward;
            }

            // REINFORCE需要回合结束时学习
            if (isReinforce) {
                ((REINFORCEAgent) agent).learnFromEpisode();
            }

            // 打印进度
            if (episode % (numEpisodes / 5) == 0) {
                System.out.printf("  Episode %d: 奖励=%.2f\n", episode, episodeReward);
            }
        }

        // 评估阶段
        if (isReinforce) {
            ((REINFORCEAgent) agent).setTraining(false);
        } else {
            ((DQNAgent) agent).setTraining(false);
        }

        float totalReward = 0.0f;
        int evalEpisodes = 10;

        for (int episode = 0; episode < evalEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;

            while (!env.isDone()) {
                Variable action;
                if (isReinforce) {
                    action = ((REINFORCEAgent) agent).selectAction(state);
                } else {
                    action = ((DQNAgent) agent).selectAction(state);
                }

                Environment.StepResult result = env.step(action);
                state = result.getNextState();
                episodeReward += result.getReward();
            }

            totalReward += episodeReward;
        }

        return totalReward / evalEpisodes;
    }

    /**
     * 打印统计信息
     *
     * @param algorithmName 算法名称
     * @param results       结果列表
     */
    private static void printStatistics(String algorithmName, List<Float> results) {
        if (results.isEmpty()) return;

        float sum = 0.0f;
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;

        for (float result : results) {
            sum += result;
            min = Math.min(min, result);
            max = Math.max(max, result);
        }

        float mean = sum / results.size();

        float variance = 0.0f;
        for (float result : results) {
            variance += (result - mean) * (result - mean);
        }
        variance /= results.size();
        float std = (float) Math.sqrt(variance);

        System.out.printf("\n=== %s 统计结果 ===\n", algorithmName);
        System.out.printf("平均值: %.2f\n", mean);
        System.out.printf("标准差: %.2f\n", std);
        System.out.printf("最小值: %.2f\n", min);
        System.out.printf("最大值: %.2f\n", max);
        System.out.printf("稳定性: %s\n", std < 10 ? "稳定" : "不稳定");
    }

    /**
     * 分析结果并给出建议
     */
    private static void analyzeAndRecommend() {
        System.out.println("\n=== 算法选择建议 ===");
        System.out.println("1. 连续控制问题：建议使用策略梯度方法(REINFORCE)");
        System.out.println("2. 离散动作空间：DQN和REINFORCE都适用");
        System.out.println("3. 样本效率：DQN通常更高效（经验回放）");
        System.out.println("4. 收敛稳定性：DQN相对更稳定（目标网络）");
        System.out.println("5. 实现复杂度：REINFORCE更简单");
        System.out.println("6. 内存需求：REINFORCE更少（不需要经验回放）");

        System.out.println("\n根据具体问题选择合适的算法！");
    }
}