package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.PPOAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;

import java.util.Map;

/**
 * CartPole环境下使用PPO算法的示例
 *
 * @author leavesfly
 * @version 0.01
 *
 * 这个示例展示了如何使用PPO（Proximal Policy Optimization）算法解决CartPole问题。
 * PPO是目前最流行的策略梯度算法之一，在稳定性和样本效率上优于基础REINFORCE算法。
 *
 * 主要学习内容：
 * 1. PPO算法的核心机制：裁剪目标函数限制策略更新幅度
 * 2. 策略网络与价值网络的双网络架构
 * 3. 优势函数（Advantage Function）的计算与使用
 * 4. PPO多轮优化（ppoEpochs）对样本效率的提升
 *
 * PPO核心思想：
 *   REINFORCE: 直接用策略梯度更新，步长难以控制，可能更新过大导致崩溃
 *   PPO裁剪目标:
 *     ratio = pi_new(a|s) / pi_old(a|s)   // 新旧策略的概率比
 *     L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
 *     优势: 限制单步更新幅度，训练更稳定
 */
public class CartPolePPOExample {

    public static void main(String[] args) {
        System.out.println("=== CartPole PPO 训练示例 ===");
        System.out.println("PPO通过裁剪目标函数限制策略更新幅度，训练更加稳定");

        // 训练参数
        int numEpisodes = 1000;          // 训练回合数
        int maxStepsPerEpisode = 500;    // 每回合最大步数
        int evaluationInterval = 100;   // 评估间隔

        // 创建环境
        Environment env = new CartPoleEnvironment(12345L);

        // 创建PPO智能体
        PPOAgent agent = createPPOAgent(env);

        // 训练智能体
        trainAgent(agent, env, numEpisodes, maxStepsPerEpisode, evaluationInterval);

        // 最终评估
        System.out.println("\n=== 最终评估 ===");
        evaluateAgent(agent, env, 10);

        System.out.println("训练完成！");
    }

    /**
     * 创建PPO智能体
     *
     * @param env 环境
     * @return PPOAgent智能体
     */
    private static PPOAgent createPPOAgent(Environment env) {
        int stateDim = env.getStateDim();    // 状态维度：4
        int actionDim = env.getActionDim();  // 动作维度：2
        int[] hiddenSizes = {128, 128};      // 策略网络和价值网络共用的隐藏层结构

        float learningRate = 0.001f;   // 学习率
        float gamma = 0.99f;           // 折扣因子
        float clipEpsilon = 0.2f;      // PPO裁剪参数：限制新旧策略偏离程度
        int ppoEpochs = 4;             // 每批数据的更新轮数：提高样本效率
        int batchSize = 32;            // mini-batch大小

        System.out.println("创建PPO智能体...");
        System.out.println("状态维度: " + stateDim);
        System.out.println("动作维度: " + actionDim);
        System.out.println("网络结构: " + stateDim + " -> " + hiddenSizes[0] + " -> " + hiddenSizes[1] + " -> " + actionDim);
        System.out.println("裁剪参数 clipEpsilon: " + clipEpsilon);
        System.out.println("PPO更新轮数 ppoEpochs: " + ppoEpochs);
        System.out.println("双网络: 策略网络(输出动作概率) + 价值网络(估计状态价值)");

        return new PPOAgent(
                "CartPole_PPO",
                stateDim, actionDim, hiddenSizes,
                learningRate, gamma, clipEpsilon,
                ppoEpochs, batchSize
        );
    }

    /**
     * 训练智能体
     *
     * @param agent              智能体
     * @param env                环境
     * @param numEpisodes        训练回合数
     * @param maxStepsPerEpisode 每回合最大步数
     * @param evaluationInterval 评估间隔
     */
    private static void trainAgent(PPOAgent agent, Environment env, int numEpisodes,
                                   int maxStepsPerEpisode, int evaluationInterval) {
        System.out.println("\n开始训练...");

        for (int episode = 0; episode < numEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;

            // 回合内收集经验
            for (int step = 0; step < maxStepsPerEpisode; step++) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);
                Variable nextState = result.getNextState();
                float reward = result.getReward();
                boolean done = result.isDone();

                // 存储经验（PPO在回合结束后统一更新）
                Experience experience = new Experience(state, action, reward, nextState, done, step);
                agent.learn(experience);

                state = nextState;
                episodeReward += reward;
                steps++;

                if (done) {
                    break;
                }
            }

            // 回合结束时执行PPO更新
            agent.learnFromEpisode();

            // 打印训练进度
            if (episode % 50 == 0 || episode == numEpisodes - 1) {
                Map<String, Object> stats = agent.getTrainingStats();
                System.out.printf("Episode %d: 奖励=%.2f, 步数=%d, 平均回报=%.2f, 策略损失=%.6f, 价值损失=%.6f\n",
                        episode, episodeReward, steps,
                        (Float) stats.get("average_return"),
                        (Float) stats.get("average_policy_loss"),
                        (Float) stats.get("average_value_loss"));
            }

            // 定期评估
            if (episode > 0 && episode % evaluationInterval == 0) {
                System.out.println("\n--- 中期评估 (Episode " + episode + ") ---");
                evaluateAgent(agent, env, 5);
                System.out.println("--- 继续训练 ---\n");
            }
        }
    }

    /**
     * 评估智能体性能（贪心模式，不探索）
     *
     * @param agent                 智能体
     * @param env                   环境
     * @param numEvaluationEpisodes 评估回合数
     */
    private static void evaluateAgent(PPOAgent agent, Environment env, int numEvaluationEpisodes) {
        agent.setTraining(false);

        float totalReward = 0.0f;
        int totalSteps = 0;
        int successfulEpisodes = 0;

        for (int episode = 0; episode < numEvaluationEpisodes; episode++) {
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;

            for (int step = 0; step < 500; step++) {
                Variable action = agent.selectAction(state);
                Environment.StepResult result = env.step(action);

                state = result.getNextState();
                episodeReward += result.getReward();
                steps++;

                if (result.isDone()) {
                    break;
                }
            }

            totalReward += episodeReward;
            totalSteps += steps;

            if (episodeReward >= 450) {
                successfulEpisodes++;
            }

            System.out.printf("评估回合 %d: 奖励=%.2f, 步数=%d\n", episode + 1, episodeReward, steps);
        }

        float averageReward = totalReward / numEvaluationEpisodes;
        float averageSteps = (float) totalSteps / numEvaluationEpisodes;
        float successRate = (float) successfulEpisodes / numEvaluationEpisodes * 100;

        System.out.println("评估结果:");
        System.out.printf("  平均奖励: %.2f\n", averageReward);
        System.out.printf("  平均步数: %.2f\n", averageSteps);
        System.out.printf("  成功率: %.1f%% (%d/%d)\n", successRate, successfulEpisodes, numEvaluationEpisodes);

        agent.setTraining(true);
    }
}
