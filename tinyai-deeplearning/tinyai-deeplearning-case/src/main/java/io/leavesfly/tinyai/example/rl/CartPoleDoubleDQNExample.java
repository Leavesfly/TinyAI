package io.leavesfly.tinyai.example.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.rl.Environment;
import io.leavesfly.tinyai.rl.Experience;
import io.leavesfly.tinyai.rl.agent.DoubleDQNAgent;
import io.leavesfly.tinyai.rl.environment.CartPoleEnvironment;

import java.util.Map;

/**
 * CartPole环境下使用Double DQN算法的示例
 *
 * @author leavesfly
 * @version 0.01
 *
 * 这个示例展示了如何使用Double DQN算法来解决CartPole（倒立摆）问题。
 * Double DQN是对标准DQN的重要改进，通过解耦动作选择和Q值评估来减少过估计问题。
 *
 * 主要学习内容：
 * 1. Double DQN算法与标准DQN的核心区别
 * 2. 如何使用DoubleDQNAgent进行训练
 * 3. 在线网络与目标网络的协同工作方式
 * 4. 训练过程的监控和性能评估
 *
 * Double DQN核心思想：
 *   标准DQN目标Q值: y = r + gamma * max_a' Q_target(s', a')
 *     问题: 同一个目标网络既选动作又评估Q值，导致系统性高估
 *   Double DQN目标Q值:
 *     a_best = argmax_a' Q_online(s', a')   // 在线网络选动作
 *     y = r + gamma * Q_target(s', a_best)  // 目标网络评估Q值
 *     优势: 分离选择与评估，减少Q值过估计
 */
public class CartPoleDoubleDQNExample {

    public static void main(String[] args) {
        System.out.println("=== CartPole Double DQN 训练示例 ===");
        System.out.println("Double DQN通过解耦动作选择和Q值评估来改进标准DQN");

        // 训练参数
        int numEpisodes = 1000;          // 训练回合数
        int maxStepsPerEpisode = 500;    // 每回合最大步数
        int evaluationInterval = 100;    // 评估间隔

        // 创建环境
        Environment env = new CartPoleEnvironment(12345L); // 使用固定种子保证可重现性

        // 创建Double DQN智能体
        DoubleDQNAgent agent = createDoubleDQNAgent(env);

        // 训练智能体
        trainAgent(agent, env, numEpisodes, maxStepsPerEpisode, evaluationInterval);

        // 最终评估
        System.out.println("\n=== 最终评估 ===");
        evaluateAgent(agent, env, 10);

        System.out.println("训练完成！");
    }

    /**
     * 创建Double DQN智能体
     *
     * @param env 环境
     * @return DoubleDQNAgent智能体
     */
    private static DoubleDQNAgent createDoubleDQNAgent(Environment env) {
        // 网络参数
        int stateDim = env.getStateDim();           // 状态维度：4
        int actionDim = env.getActionDim();         // 动作维度：2
        int[] hiddenSizes = {128, 128};             // 隐藏层尺寸

        // 算法参数（与DQN相同，便于对比）
        float learningRate = 0.001f;                // 学习率
        float epsilon = 1.0f;                       // 初始探索率
        float gamma = 0.99f;                        // 折扣因子
        int batchSize = 32;                         // 批次大小
        int bufferSize = 10000;                     // 经验回放缓冲区大小
        int targetUpdateFreq = 100;                 // 目标网络更新频率

        System.out.println("创建Double DQN智能体...");
        System.out.println("状态维度: " + stateDim);
        System.out.println("动作维度: " + actionDim);
        System.out.println("网络结构: " + stateDim + " -> " + hiddenSizes[0] + " -> " + hiddenSizes[1] + " -> " + actionDim);
        System.out.println("学习率: " + learningRate);
        System.out.println("初始探索率: " + epsilon);
        System.out.println("Double DQN改进: 在线网络选动作，目标网络评估Q值");

        return new DoubleDQNAgent(
                "CartPole_DoubleDQN",
                stateDim, actionDim, hiddenSizes,
                learningRate, epsilon, gamma,
                batchSize, bufferSize, targetUpdateFreq
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
    private static void trainAgent(DoubleDQNAgent agent, Environment env, int numEpisodes,
                                   int maxStepsPerEpisode, int evaluationInterval) {
        System.out.println("\n开始训练...");

        for (int episode = 0; episode < numEpisodes; episode++) {
            // 重置环境
            Variable state = env.reset();
            float episodeReward = 0.0f;
            int steps = 0;

            for (int step = 0; step < maxStepsPerEpisode; step++) {
                // 选择动作
                Variable action = agent.selectAction(state);

                // 执行动作
                Environment.StepResult result = env.step(action);
                Variable nextState = result.getNextState();
                float reward = result.getReward();
                boolean done = result.isDone();

                // 存储经验并学习
                Experience experience = new Experience(state, action, reward, nextState, done, step);
                agent.learn(experience);

                // 更新状态和累积奖励
                state = nextState;
                episodeReward += reward;
                steps++;

                if (done) {
                    break;
                }
            }

            // 打印训练进度
            if (episode % 50 == 0 || episode == numEpisodes - 1) {
                Map<String, Object> stats = agent.getTrainingStats();
                System.out.printf("Episode %d: 奖励=%.2f, 步数=%d, Epsilon=%.3f, 损失=%.6f, 缓冲区使用率=%.2f%%\n",
                        episode, episodeReward, steps,
                        (Float) stats.get("epsilon"),
                        (Float) stats.get("average_loss"),
                        (Float) stats.get("buffer_usage") * 100);
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
     * 评估智能体性能
     *
     * @param agent                 智能体
     * @param env                   环境
     * @param numEvaluationEpisodes 评估回合数
     */
    private static void evaluateAgent(DoubleDQNAgent agent, Environment env, int numEvaluationEpisodes) {
        // 切换到评估模式
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

            if (episodeReward >= 450) { // 认为450+步为成功
                successfulEpisodes++;
            }

            System.out.printf("评估回合 %d: 奖励=%.2f, 步数=%d\n", episode + 1, episodeReward, steps);
        }

        // 计算平均性能
        float averageReward = totalReward / numEvaluationEpisodes;
        float averageSteps = (float) totalSteps / numEvaluationEpisodes;
        float successRate = (float) successfulEpisodes / numEvaluationEpisodes * 100;

        System.out.println("评估结果:");
        System.out.printf("  平均奖励: %.2f\n", averageReward);
        System.out.printf("  平均步数: %.2f\n", averageSteps);
        System.out.printf("  成功率: %.1f%% (%d/%d)\n", successRate, successfulEpisodes, numEvaluationEpisodes);

        // 切换回训练模式
        agent.setTraining(true);
    }
}
