package io.leavesfly.tinyai.ml.optimize;

/**
 * 学习率调度器接口，提供灵活的学习率调整策略，用于训练过程中动态调整优化器学习率。
 * 类似 PyTorch 的 lr_scheduler。
 *
 * @author TinyAI
 * @version 2.0
 */
public interface LRScheduler {

    /**
     * 根据当前训练步数获取学习率
     *
     * @param step 当前训练步数
     * @return 学习率
     */
    float getLearningRate(int step);

    /**
     * 常数学习率调度器
     */
    class ConstantLR implements LRScheduler {
        private final float lr;

        public ConstantLR(float lr) {
            this.lr = lr;
        }

        @Override
        public float getLearningRate(int step) {
            return lr;
        }
    }

    /**
     * 阶梯衰减学习率调度器，每隔 stepSize 步，学习率乘以 gamma。
     */
    class StepLR implements LRScheduler {
        private final float baseLR;
        private final int stepSize;
        private final float gamma;

        /**
         * 构造函数
         *
         * @param baseLR   基础学习率
         * @param stepSize 衰减步长
         * @param gamma    衰减因子
         */
        public StepLR(float baseLR, int stepSize, float gamma) {
            this.baseLR = baseLR;
            this.stepSize = stepSize;
            this.gamma = gamma;
        }

        @Override
        public float getLearningRate(int step) {
            int epochs = step / stepSize;
            return (float) (baseLR * Math.pow(gamma, epochs));
        }
    }

    /**
     * 余弦退火学习率调度器，使用余弦函数平滑降低学习率。
     */
    class CosineAnnealingLR implements LRScheduler {
        private final float baseLR;
        private final float minLR;
        private final int T_max;

        /**
         * 构造函数
         *
         * @param baseLR 基础学习率
         * @param minLR  最小学习率
         * @param T_max  最大步数
         */
        public CosineAnnealingLR(float baseLR, float minLR, int T_max) {
            this.baseLR = baseLR;
            this.minLR = minLR;
            this.T_max = T_max;
        }

        @Override
        public float getLearningRate(int step) {
            if (step >= T_max) {
                return minLR;
            }
            float cosineDecay = (float) (0.5 * (1 + Math.cos(Math.PI * step / T_max)));
            return minLR + (baseLR - minLR) * cosineDecay;
        }
    }

    /**
     * 指数衰减学习率调度器
     */
    class ExponentialLR implements LRScheduler {
        private final float baseLR;
        private final float gamma;

        /**
         * 构造函数
         *
         * @param baseLR 基础学习率
         * @param gamma  衰减因子
         */
        public ExponentialLR(float baseLR, float gamma) {
            this.baseLR = baseLR;
            this.gamma = gamma;
        }

        @Override
        public float getLearningRate(int step) {
            return (float) (baseLR * Math.pow(gamma, step));
        }
    }

    /**
     * 多步衰减学习率调度器，在指定的步数处降低学习率。
     */
    class MultiStepLR implements LRScheduler {
        private final float baseLR;
        private final int[] milestones;
        private final float gamma;

        /**
         * 构造函数
         *
         * @param baseLR     基础学习率
         * @param milestones 衰减步数里程碑
         * @param gamma      衰减因子
         */
        public MultiStepLR(float baseLR, int[] milestones, float gamma) {
            this.baseLR = baseLR;
            this.milestones = milestones;
            this.gamma = gamma;
        }

        @Override
        public float getLearningRate(int step) {
            int count = 0;
            for (int milestone : milestones) {
                if (step >= milestone) {
                    count++;
                }
            }
            return (float) (baseLR * Math.pow(gamma, count));
        }
    }

    /**
     * 线性预热学习率调度器，在 warmup_steps 内线性增加学习率，之后保持不变。
     */
    class LinearWarmupLR implements LRScheduler {
        private final float baseLR;
        private final int warmupSteps;

        /**
         * 构造函数
         *
         * @param baseLR      目标学习率
         * @param warmupSteps 预热步数
         */
        public LinearWarmupLR(float baseLR, int warmupSteps) {
            this.baseLR = baseLR;
            this.warmupSteps = warmupSteps;
        }

        @Override
        public float getLearningRate(int step) {
            if (step < warmupSteps) {
                return baseLR * (step + 1) / warmupSteps;
            }
            return baseLR;
        }
    }
}
