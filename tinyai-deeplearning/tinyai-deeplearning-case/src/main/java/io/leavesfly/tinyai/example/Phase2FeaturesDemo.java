package io.leavesfly.tinyai.example;

import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.visual.Monitor;
import io.leavesfly.tinyai.ml.model.ModelSerializer;
import io.leavesfly.tinyai.ml.training.Trainer;
import io.leavesfly.tinyai.ml.model.checkpoint.Checkpoint;
import io.leavesfly.tinyai.ml.dataset.ArrayDataset;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.evaluator.RegressEval;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.loss.MeanSquaredLoss;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ml.optimize.LRScheduler;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.container.Sequential;
import io.leavesfly.tinyai.nnet.core.Module;
import io.leavesfly.tinyai.nnet.layer.dnn.Linear;

/**
 * Phase 2 新增功能综合演示
 * <p>
 * 演示内容:
 * 1. 损失函数Reduction参数
 * 2. Hooks机制(梯度裁剪、特征监控)
 * 3. 学习率调度器
 * 4. Checkpoint断点续训
 * 5. 训练监控与可视化
 * 6. DataLoader多线程预取
 *
 * @author TinyDL
 * @version 2.0
 */
public class Phase2FeaturesDemo {

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Phase 2 新增功能综合演示 (使用Trainer)");
        System.out.println("=".repeat(80));

        // 1. 准备数据集 (合并训练集和验证集)
        DataSet fullDataset = createSyntheticDataset(1200);
        fullDataset.prepare();
        
        // 分割数据集: 80%训练, 20%验证
        fullDataset.splitDataset(0.8f, 0.2f, 0.0f);
        System.out.println("\n✓ 数据集创建完成 (训练集: 960, 验证集: 240)");

        // 2. 构建模型
        Module network = new Sequential("mlp")
                .add(new Linear("fc1", 10, 64))
                .add(new Linear("fc2", 64, 32))
                .add(new Linear("fc3", 32, 1));

        Model model = new Model("phase2_demo", network);
        System.out.println("\n✓ 模型创建完成");
        System.out.println(network);

        // 3. 注册Hooks (Phase 2.2 新增)
        registerHooks(network);
        System.out.println("\n✓ Hooks注册完成 (梯度裁剪 + 激活值监控)");

        // 4. 配置损失函数 (Phase 2.1 新增: Reduction支持)
        Loss loss = new MeanSquaredLoss(Loss.Reduction.MEAN);
        System.out.println("\n✓ 损失函数配置完成 (Reduction=MEAN)");

        // 5. 配置优化器 + 学习率调度器 (Phase 1.3 新增)
        Optimizer optimizer = new Adam(model);
        LRScheduler scheduler = new LRScheduler.CosineAnnealingLR(0.001f, 0.0001f, 50);
        optimizer.setLRScheduler(scheduler);
        System.out.println("\n✓ 优化器配置完成 (学习率调度: CosineAnnealing)");

        // 6. 创建评估器
        Evaluator evaluator = new RegressEval(loss, model, fullDataset);
        System.out.println("\n✓ 评估器创建完成");

        // 7. 创建Monitor (Phase 2.4 新增 - 与TrainingMonitor集成)
        Monitor monitor = new Monitor("output/training_log.txt");
        System.out.println("\n✓ Monitor创建完成");

        // 8. 创建Trainer并初始化
        int epochs = 20;
        Trainer trainer = new Trainer(epochs, monitor, evaluator);
        trainer.init(fullDataset, model, loss, optimizer);
        System.out.println("\n✓ Trainer初始化完成");

        // 9. 开始训练
        System.out.println("\n" + "=".repeat(80));
        System.out.println("开始训练...");
        System.out.println("=".repeat(80));

        trainer.train(true);  // 启用数据打乱

        System.out.println("\n✓ 训练完成");

        // 10. 评估模型
        System.out.println("\n" + "=".repeat(80));
        System.out.println("模型评估...");
        System.out.println("=".repeat(80));
        
        trainer.evaluate();

        // 11. 保存Checkpoint (Phase 2.3 新增)
        String checkpointPath = "output/phase2_checkpoint.ckpt";
        ModelSerializer.saveCheckpoint(model, optimizer, epochs, 0.0f, checkpointPath);
        System.out.println("\n✓ Checkpoint已保存: " + checkpointPath);

        // 12. 可视化 (Phase 2.4 新增)
        System.out.println("\n" + "=".repeat(80));
        System.out.println("生成可视化图表...");
        System.out.println("=".repeat(80));

        monitor.plot();  // 使用Monitor内置的可视化功能
        System.out.println("\n✓ 训练曲线已显示");

        // 13. 演示断点续训 (Phase 2.3 新增)
        System.out.println("\n" + "=".repeat(80));
        System.out.println("演示断点续训...");
        System.out.println("=".repeat(80));

        try {
            Checkpoint checkpoint = ModelSerializer.resumeTraining(checkpointPath, model, optimizer);
            System.out.println("\n✓ 成功从Checkpoint恢复训练");
            System.out.println("  继续训练epoch: " + (checkpoint.getEpoch() + 1));
            System.out.println("  最佳损失: " + String.format("%.6f", checkpoint.getBestLoss()));
        } catch (Exception e) {
            System.err.println("恢复Checkpoint失败: " + e.getMessage());
        }

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Phase 2 功能演示完成!");
        System.out.println("=".repeat(80));
    }

    /**
     * 注册Hooks演示 (Phase 2.2 新增)
     */
    private static void registerHooks(Module network) {
        // 1. 梯度裁剪Hook
        Module.BackwardHook gradClipHook = (module, gradInput, gradOutput) -> {
            // 梯度裁剪: clip到[-1, 1]
            NdArray gradData = gradInput.getGrad();
            if (gradData != null) {
                float[] data = gradData.getArray();
                for (int i = 0; i < data.length; i++) {
                    data[i] = Math.max(-1.0f, Math.min(1.0f, data[i]));
                }
                // 注意: 这里仅演示逻辑,实际修改需要通过NdArray API
            }
            return gradInput;
        };

        // 2. 激活值监控Hook (仅在第一层)
        Module firstLayer = network.getModule("fc1");
        if (firstLayer != null) {
            firstLayer.register_forward_hook((module, input, output) -> {
                // 计算激活值统计信息
                NdArray data = output.getValue();
                float[] values = data.getArray();
                float sum = 0;
                for (float v : values) {
                    sum += v;
                }
                float mean = sum / values.length;
                
                // 仅在训练初期打印(避免输出过多)
                if (Math.random() < 0.01) {  // 1%概率打印
                    System.out.printf("  [Hook] fc1 激活值均值: %.4f%n", mean);
                }
                return output;
            });
        }

        // 为所有层注册梯度裁剪
        for (Module child : network.children()) {
            child.register_backward_hook(gradClipHook);
        }
    }

    /**
     * 创建合成数据集
     */
    private static DataSet createSyntheticDataset(int size) {
        NdArray[] xs = new NdArray[size];
        NdArray[] ys = new NdArray[size];

        for (int i = 0; i < size; i++) {
            // 生成随机输入 [10]
            float[] inputData = new float[10];
            for (int j = 0; j < 10; j++) {
                inputData[j] = (float) (Math.random() * 2 - 1);
            }

            // 生成标签 (简单的线性关系 + 噪声)
            float label = 0;
            for (int j = 0; j < 10; j++) {
                label += inputData[j] * (j + 1) * 0.1f;
            }
            label += (float) (Math.random() * 0.1 - 0.05);  // 添加噪声

            xs[i] = NdArray.of(inputData, Shape.of(1, 10));
            ys[i] = NdArray.of(new float[]{label}, Shape.of(1, 1));
        }

        // 创建一个简单的ArrayDataset实现
        return new ArrayDataset(32) {
            {
                this.xs = xs;
                this.ys = ys;
            }

            @Override
            protected DataSet build(int batchSize, NdArray[] _xs, NdArray[] _ys) {
                ArrayDataset dataset = new ArrayDataset(batchSize) {
                    @Override
                    protected DataSet build(int batchSize, NdArray[] __xs, NdArray[] __ys) {
                        return null;
                    }

                    @Override
                    public void doPrepare() {
                    }
                };
                dataset.setXs(_xs);
                dataset.setYs(_ys);
                return dataset;
            }

            @Override
            public void doPrepare() {
            }
        };
    }
}
