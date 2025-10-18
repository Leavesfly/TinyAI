package io.leavesfly.tinyai.agent.vla.env;

import io.leavesfly.tinyai.agent.vla.model.VLAState;
import io.leavesfly.tinyai.agent.vla.model.VLAAction;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Map;

/**
 * 机器人环境接口
 * 定义VLA智能体与环境交互的统一接口
 *
 * @author TinyAI
 */
public interface RobotEnvironment {

    /**
     * 重置环境到初始状态
     *
     * @return 初始观测状态
     */
    VLAState reset();

    /**
     * 执行动作并返回新状态
     *
     * @param action VLA动作
     * @return 环境反馈（新状态、奖励、终止标志、额外信息）
     */
    EnvironmentStep step(VLAAction action);

    /**
     * 渲染当前环境状态（可选）
     *
     * @return 渲染图像
     */
    byte[] render();

    /**
     * 获取动作空间规格
     */
    ActionSpaceSpec getActionSpace();

    /**
     * 获取观测空间规格
     */
    ObservationSpaceSpec getObservationSpace();

    /**
     * 关闭环境，释放资源
     */
    void close();

    /**
     * 采样随机动作
     *
     * @return 随机动作
     */
    default VLAAction sampleAction() {
        // 默认实现：返回零动作
        return new VLAAction(NdArray.zeros(Shape.of(1, 2, 3)));
    }

    /**
     * 环境步骤返回值
     */
    class EnvironmentStep {
        private VLAState nextState;
        private double reward;
        private boolean done;
        private Map<String, Object> info;

        public EnvironmentStep(VLAState nextState, double reward, boolean done, Map<String, Object> info) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }

        public VLAState getNextState() {
            return nextState;
        }

        public double getReward() {
            return reward;
        }

        public boolean isDone() {
            return done;
        }

        public Map<String, Object> getInfo() {
            return info;
        }
    }

    /**
     * 动作空间规格
     */
    class ActionSpaceSpec {
        private int continuousDim;
        private int discreteNum;
        private double[] continuousLow;
        private double[] continuousHigh;

        public ActionSpaceSpec(int continuousDim, int discreteNum,
                               double[] continuousLow, double[] continuousHigh) {
            this.continuousDim = continuousDim;
            this.discreteNum = discreteNum;
            this.continuousLow = continuousLow;
            this.continuousHigh = continuousHigh;
        }

        public int getContinuousDim() {
            return continuousDim;
        }

        public int getDiscreteNum() {
            return discreteNum;
        }

        public double[] getContinuousLow() {
            return continuousLow;
        }

        public double[] getContinuousHigh() {
            return continuousHigh;
        }
    }

    /**
     * 观测空间规格
     */
    class ObservationSpaceSpec {
        private int imageHeight;
        private int imageWidth;
        private int imageChannels;
        private int proprietaryDim;

        public ObservationSpaceSpec(int imageHeight, int imageWidth, int imageChannels, int proprietaryDim) {
            this.imageHeight = imageHeight;
            this.imageWidth = imageWidth;
            this.imageChannels = imageChannels;
            this.proprietaryDim = proprietaryDim;
        }

        public int getImageHeight() {
            return imageHeight;
        }

        public int getImageWidth() {
            return imageWidth;
        }

        public int getImageChannels() {
            return imageChannels;
        }

        public int getProprietaryDim() {
            return proprietaryDim;
        }
    }
}
