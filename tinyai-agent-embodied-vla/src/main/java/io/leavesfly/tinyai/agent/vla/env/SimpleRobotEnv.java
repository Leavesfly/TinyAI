package io.leavesfly.tinyai.agent.vla.env;

import io.leavesfly.tinyai.agent.vla.model.*;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 简单机器人仿真环境
 * 实现基础的PickAndPlace任务场景
 *
 * @author TinyAI
 */
public class SimpleRobotEnv implements RobotEnvironment {

    private final int imageSize;
    private final int numJoints;
    private final TaskConfig config;
    private final Random random;

    // 环境状态
    private float[] jointPositions;
    private float[] jointVelocities;
    private float[] endEffectorPos;
    private float gripperState;
    private float[] objectPos;
    private float[] targetPos;

    private int currentStep;
    private boolean objectGrasped;

    /**
     * 构造函数
     */
    public SimpleRobotEnv(TaskConfig config) {
        this.config = config;
        this.imageSize = 64;
        this.numJoints = 7;
        this.random = new Random(config.getRandomSeed());

        reset();
    }

    @Override
    public VLAState reset() {
        // 初始化关节状态
        jointPositions = new float[numJoints];
        jointVelocities = new float[numJoints];

        for (int i = 0; i < numJoints; i++) {
            jointPositions[i] = (float) ((random.nextFloat() - 0.5) * 0.2);
            jointVelocities[i] = 0.0F;
        }

        // 初始化末端执行器位置
        endEffectorPos = new float[]{0.5f, 0.0f, 0.3f};
        gripperState = 0.0f; // 打开

        // 随机放置物体
        objectPos = new float[]{
                (float) (0.3 + random.nextDouble() * 0.3),
                (float) (-0.3 + random.nextDouble() * 0.6),
                0.0f
        };

        // 固定目标位置
        targetPos = new float[]{0.5f, 0.5f, 0.0f};

        currentStep = 0;
        objectGrasped = false;

        return getCurrentState();
    }

    @Override
    public EnvironmentStep step(VLAAction action) {
        // 更新末端执行器位置（简化模拟）
        NdArray continuousAction = action.getContinuousAction();
        if (continuousAction != null) {
            float[] actionValues = continuousAction.getArray();

            // 更新末端执行器位置 (前3个维度是位置增量)
            for (int i = 0; i < 3 && i < actionValues.length; i++) {
                endEffectorPos[i] += actionValues[i] * 0.01; // 缩放增量
                endEffectorPos[i] = (float) Math.max(-1.0, Math.min(1.0, endEffectorPos[i]));
            }

            // 更新夹爪状态 (最后一个维度)
            if (actionValues.length > 6) {
                gripperState = (float) ((actionValues[6] + 1.0) / 2.0); // 映射到[0, 1]
            }
        }

        // 检查抓取
        if (!objectGrasped && gripperState > 0.5) {
            double dist = distance(endEffectorPos, objectPos);
            if (dist < 0.05) {
                objectGrasped = true;
            }
        }

        // 如果抓取了物体，物体跟随末端执行器
        if (objectGrasped) {
            objectPos[0] = endEffectorPos[0];
            objectPos[1] = endEffectorPos[1];
            objectPos[2] = endEffectorPos[2];
        }

        // 计算奖励
        double reward = calculateReward();

        // 检查终止条件
        boolean done = checkDone();

        // 增加步数
        currentStep++;

        // 构造下一个状态
        VLAState nextState = getCurrentState();

        // 额外信息
        Map<String, Object> info = new HashMap<>();
        info.put("object_grasped", objectGrasped);
        info.put("distance_to_target", distance(objectPos, targetPos));
        info.put("current_step", currentStep);

        return new EnvironmentStep(nextState, reward, done, info);
    }

    /**
     * 计算奖励
     */
    private double calculateReward() {
        double reward = 0.0;

        // 奖励1：接近物体
        if (!objectGrasped) {
            double distToObject = distance(endEffectorPos, objectPos);
            reward -= distToObject;
        }

        // 奖励2：抓取成功
        if (objectGrasped) {
            reward += 10.0;
        }

        // 奖励3：将物体放到目标位置
        double distToTarget = distance(objectPos, targetPos);
        reward -= distToTarget;

        if (objectGrasped && distToTarget < 0.05) {
            reward += 50.0; // 任务完成
        }

        return reward;
    }

    /**
     * 检查是否终止
     */
    private boolean checkDone() {
        // 达到最大步数
        if (currentStep >= config.getMaxSteps()) {
            return true;
        }

        // 任务完成
        if (objectGrasped && distance(objectPos, targetPos) < 0.05) {
            return true;
        }

        return false;
    }

    /**
     * 获取当前状态
     */
    private VLAState getCurrentState() {
        // 创建虚拟图像（简化）
        double[][][] rgbImage = new double[imageSize][imageSize][3];
        for (int h = 0; h < imageSize; h++) {
            for (int w = 0; w < imageSize; w++) {
                rgbImage[h][w][0] = random.nextDouble() * 0.1; // 背景
                rgbImage[h][w][1] = random.nextDouble() * 0.1;
                rgbImage[h][w][2] = random.nextDouble() * 0.1;
            }
        }

        VisionInput visionInput = new VisionInput(NdArray.of(rgbImage));

        // 创建语言输入（任务指令）
        LanguageInput languageInput = new LanguageInput(
                "Pick up the object and place it at the target position"
        );

        // 创建本体感知输入
        ProprioceptionInput proprioInput = new ProprioceptionInput(
                NdArray.of(jointPositions),
                NdArray.of(jointVelocities),
                NdArray.of(endEffectorPos),
                gripperState
        );

        return new VLAState(visionInput, languageInput, proprioInput);
    }

    /**
     * 计算欧氏距离
     */
    private double distance(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    @Override
    public byte[] render() {
        // 简化实现：返回空数组
        return new byte[0];
    }

    @Override
    public ActionSpaceSpec getActionSpace() {
        double[] low = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
        double[] high = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        return new ActionSpaceSpec(7, 7, low, high);
    }

    @Override
    public ObservationSpaceSpec getObservationSpace() {
        return new ObservationSpaceSpec(imageSize, imageSize, 3, numJoints * 2 + 1);
    }

    @Override
    public void close() {
        // 清理资源
    }
}
