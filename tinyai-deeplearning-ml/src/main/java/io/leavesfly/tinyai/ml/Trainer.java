package io.leavesfly.tinyai.ml;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.parallel.GradientAggregator;
import io.leavesfly.tinyai.ml.parallel.ParallelBatchProcessor;
import io.leavesfly.tinyai.ml.parallel.ParallelTrainingUtils;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * 模型训练器
 * <p>
 * 该类是TinyDL框架中模型训练的核心组件，提供了完整的训练流程管理功能，
 * 支持单线程和并行训练两种模式。
 * <p>
 * 主要功能：
 * 1. 训练流程管理：控制训练的轮次、批次处理等
 * 2. 单线程训练：传统的顺序训练模式
 * 3. 并行训练：支持多线程并行处理批次数据
 * 4. 训练监控：与Monitor配合收集训练过程信息
 * 5. 模型评估：与Evaluator配合进行模型性能评估
 *
 * @author TinyDL
 * @version 1.0
 */
public class Trainer {

    private DataSet dataSet;

    private Model model;

    private Loss loss;

    private Optimizer optimizer;

    private Monitor monitor;

    private Evaluator evaluator;

    private int maxEpoch;

    // 并行训练相关配置
    private int parallelThreadCount;
    private ExecutorService executorService;
    private boolean enableParallelTraining;

    /**
     * 构造器（默认不启用并行训练）
     *
     * @param _maxEpoch  最大训练轮次
     * @param _monitor   监控器
     * @param _evaluator 评估器
     */
    public Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator) {
        this.maxEpoch = _maxEpoch;
        monitor = _monitor;
        evaluator = _evaluator;

        // 默认并行训练配置
        this.enableParallelTraining = false;
        this.parallelThreadCount = ParallelTrainingUtils.getRecommendedThreadCount(4); // 默认根据4个batch计算
    }

    /**
     * 构造器 - 支持并行训练配置
     *
     * @param _maxEpoch      最大训练轮次
     * @param _monitor       监控器
     * @param _evaluator     评估器
     * @param enableParallel 是否启用并行训练
     * @param threadCount    并行线程数（0表示自动计算）
     */
    public Trainer(int _maxEpoch, Monitor _monitor, Evaluator _evaluator,
                   boolean enableParallel, int threadCount) {
        this.maxEpoch = _maxEpoch;
        monitor = _monitor;
        evaluator = _evaluator;
        this.enableParallelTraining = enableParallel;
        this.parallelThreadCount = threadCount > 0 ? threadCount :
                ParallelTrainingUtils.getRecommendedThreadCount(4);
    }

    /**
     * 初始化训练器
     *
     * @param _dataSet   数据集
     * @param _model     模型
     * @param _loss      损失函数
     * @param _optimizer 优化器
     */
    public void init(DataSet _dataSet, Model _model, Loss _loss, Optimizer _optimizer) {
        dataSet = _dataSet;
        _dataSet.prepare();

        model = _model;
        loss = _loss;
        optimizer = _optimizer;

        // 检查模型是否支持并行训练
        if (enableParallelTraining && !ParallelTrainingUtils.isModelParallelizable(model)) {
            System.err.println("警告: 模型不支持并行训练，将回退到单线程模式");
            enableParallelTraining = false;
        }

        // 初始化线程池
        if (enableParallelTraining) {
            // 根据实际batch数重新计算线程数
            DataSet trainDataSet = dataSet.getTrainDataSet();
            if (trainDataSet != null) {
                List<Batch> batches = trainDataSet.getBatches();
                parallelThreadCount = Math.min(parallelThreadCount, batches.size());
            }

            executorService = Executors.newFixedThreadPool(parallelThreadCount);
            System.out.println("并行训练已启用，线程数: " + parallelThreadCount);
        }
    }

    /**
     * 主训练方法 - 自动选择单线程或并行训练
     *
     * @param shuffleData 是否打乱数据
     */
    public void train(boolean shuffleData) {
        if (enableParallelTraining) {
            parallelTrain(shuffleData);
        } else {
            singleThreadTrain(shuffleData);
        }
    }

    /**
     * 单线程训练（原始实现）
     *
     * @param shuffleData 是否打乱数据
     */
    public void singleThreadTrain(boolean shuffleData) {

        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }

        for (int i = 0; i < maxEpoch; i++) {

            model.resetState();
            monitor.startNewEpoch(i);

            List<Batch> batches = trainDataSet.getBatches();
            float lossSum = 0f;
            float accSum = 0f;

            for (Batch batch : batches) {
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

                Variable predictY = model.forward(variableX);
                Variable lossVariable = loss.loss(variableY, predictY);
                lossVariable.setName("loss");

                model.clearGrads();
                lossSum += lossVariable.getValue().getNumber().floatValue();

                lossVariable.backward();

                optimizer.update();
                lossVariable.unChainBackward();

                model.tmpPredict = predictY;
            }
            monitor.collectInfo(lossSum / batches.size());
            monitor.endEpoch();
            monitor.printTrainInfo();
        }
        monitor.plot();
    }

    /**
     * 并行训练实现
     * 将batch分配给多个线程并行处理，然后聚合梯度并更新参数
     *
     * @param shuffleData 是否打乱数据
     */
    public void parallelTrain(boolean shuffleData) {
        if (!enableParallelTraining || executorService == null) {
            System.err.println("警告: 并行训练未启用，回退到单线程模式");
            singleThreadTrain(shuffleData);
            return;
        }

        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }

        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            long epochStartTime = System.currentTimeMillis();

            model.resetState();
            monitor.startNewEpoch(epoch);

            List<Batch> batches = trainDataSet.getBatches();

            // 检查是否有足够的batch进行并行处理
            if (batches.size() < parallelThreadCount) {
                // 如果batch数量少于线程数，使用单线程处理
                processBatchesSequentially(batches, epoch);
            } else {
                // 使用并行处理
                processBatchesInParallel(batches, epoch);
            }

            long epochEndTime = System.currentTimeMillis();
            monitor.endEpoch();
            System.out.println(String.format("Epoch %d 完成，耗时: %d ms",
                    epoch, epochEndTime - epochStartTime));
        }

        monitor.plot();
    }

    /**
     * 并行处理批次数据
     *
     * @param batches 批次列表
     * @param epoch   当前轮次
     */
    private void processBatchesInParallel(List<Batch> batches, int epoch) {
        int batchCount = batches.size();
        float totalLoss = 0f;
        int successfulBatches = 0;

        // 按线程数分组处理batch
        for (int i = 0; i < batchCount; i += parallelThreadCount) {
            int endIndex = Math.min(i + parallelThreadCount, batchCount);
            List<Batch> currentBatchGroup = batches.subList(i, endIndex);

            // 为这一组batch创建梯度聚合器
            GradientAggregator gradientAggregator = new GradientAggregator(currentBatchGroup.size());

            // 提交并行任务
            List<Future<ParallelBatchProcessor.BatchProcessResult>> futures = new ArrayList<>();

            for (int j = 0; j < currentBatchGroup.size(); j++) {
                Batch batch = currentBatchGroup.get(j);
                Model modelCopy = ParallelTrainingUtils.deepCopyModel(model);

                ParallelBatchProcessor processor = new ParallelBatchProcessor(
                        batch, modelCopy, loss, gradientAggregator, i + j
                );

                futures.add(executorService.submit(processor));
            }

            // 收集结果
            float groupLoss = 0f;
            int groupSuccessful = 0;

            for (Future<ParallelBatchProcessor.BatchProcessResult> future : futures) {
                try {
                    ParallelBatchProcessor.BatchProcessResult result = future.get();
                    if (result.isSuccess()) {
                        groupLoss += result.getLossValue();
                        groupSuccessful++;
                    } else {
                        System.err.println("批次处理失败: " + result.getException().getMessage());
                    }
                } catch (Exception e) {
                    System.err.println("获取批次处理结果失败: " + e.getMessage());
                }
            }

            // 等待梯度聚合完成
            try {
                Map<String, NdArray> averageGradients = gradientAggregator.getAverageGradients();

                // 将聚合梯度应用到主模型
                ParallelTrainingUtils.applyAggregatedGradients(model, averageGradients);

                // 更新参数
                optimizer.update();

                // 清理梯度
                model.clearGrads();

            } catch (InterruptedException e) {
                System.err.println("梯度聚合被中断: " + e.getMessage());
                Thread.currentThread().interrupt();
                break;
            }

            totalLoss += groupLoss;
            successfulBatches += groupSuccessful;
        }

        // 更新监控信息
        if (successfulBatches > 0) {
            monitor.collectInfo(totalLoss / successfulBatches);
        }
        monitor.endEpoch();
        monitor.printTrainInfo();
    }

    /**
     * 顺序处理批次数据（备用方案）
     *
     * @param batches 批次列表
     * @param epoch   当前轮次
     */
    private void processBatchesSequentially(List<Batch> batches, int epoch) {
        float lossSum = 0f;

        for (Batch batch : batches) {
            Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
            Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

            Variable predictY = model.forward(variableX);
            Variable lossVariable = loss.loss(variableY, predictY);
            lossVariable.setName("loss");

            model.clearGrads();
            lossSum += lossVariable.getValue().getNumber().floatValue();

            lossVariable.backward();
            optimizer.update();
            lossVariable.unChainBackward();

            model.tmpPredict = predictY;
        }

        monitor.collectInfo(lossSum / batches.size());
        monitor.printTrainInfo();
    }


    /**
     * 模型评估
     */
    public void evaluate() {
        evaluator.evaluate();
    }

    /**
     * 设置并行训练参数
     *
     * @param enable      是否启用并行训练
     * @param threadCount 线程数（0表示自动计算）
     */
    public void configureParallelTraining(boolean enable, int threadCount) {
        // 先关闭现有的线程池
        if (executorService != null && !executorService.isShutdown()) {
            shutdown();
        }

        this.enableParallelTraining = enable;
        if (threadCount > 0) {
            this.parallelThreadCount = threadCount;
        }

        // 如果启用并且模型已初始化，重新创建线程池
        if (enable && model != null) {
            if (ParallelTrainingUtils.isModelParallelizable(model)) {
                executorService = Executors.newFixedThreadPool(parallelThreadCount);
                System.out.println("并行训练已重新配置，线程数: " + parallelThreadCount);
            } else {
                System.err.println("模型不支持并行训练");
                this.enableParallelTraining = false;
            }
        }
    }

    /**
     * 获取并行训练状态
     *
     * @return true 如果并行训练已启用
     */
    public boolean isParallelTrainingEnabled() {
        return enableParallelTraining && executorService != null && !executorService.isShutdown();
    }

    /**
     * 获取并行线程数
     *
     * @return 并行线程数
     */
    public int getParallelThreadCount() {
        return parallelThreadCount;
    }

    /**
     * 关闭训练器并释放资源
     * 必须在训练结束后调用此方法以防止资源泄漏
     */
    public void shutdown() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
            try {
                // 等待正在执行的任务完成
                if (!executorService.awaitTermination(30, TimeUnit.SECONDS)) {
                    // 强制停止
                    System.err.println("警告: 强制关闭线程池");
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                System.err.println("线程池关闭被中断");
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
            System.out.println("并行训练资源已释放");
        }
    }

    /**
     * 简化版并行训练实现 - 不依赖模型序列化
     * 通过批次级并行思维展示并行训练概念
     * 这是一个演示版本，适用于在模型不支持序列化时展示并行训练思路
     *
     * @param shuffleData 是否打乱数据
     */
    public void simplifiedParallelTrain(boolean shuffleData) {
        System.out.println("使用简化版并行训练演示...");

        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }

        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            long epochStartTime = System.currentTimeMillis();

            model.resetState();
            monitor.startNewEpoch(epoch);

            List<Batch> batches = trainDataSet.getBatches();

            // 模拟并行处理（实际仍是顺序处理，但显示并行思维）
            float totalLoss = 0f;
            int processedBatches = 0;

            System.out.println(String.format("处理 %d 个批次，模拟 %d 个并行线程...",
                    batches.size(), parallelThreadCount));

            for (int i = 0; i < batches.size(); i++) {
                Batch batch = batches.get(i);

                // 模拟并行处理的日志
                int threadId = i % parallelThreadCount;
                System.out.println(String.format("  [线程-%d] 处理批次 %d/%d",
                        threadId, i + 1, batches.size()));

                try {
                    Variable variableX = batch.toVariableX().setName("x_" + threadId).setRequireGrad(false);
                    Variable variableY = batch.toVariableY().setName("y_" + threadId).setRequireGrad(false);

                    Variable predictY = model.forward(variableX);
                    Variable lossVariable = loss.loss(variableY, predictY);
                    lossVariable.setName("loss_" + threadId);

                    model.clearGrads();
                    float lossValue = lossVariable.getValue().getNumber().floatValue();
                    totalLoss += lossValue;

                    lossVariable.backward();
                    optimizer.update();
                    lossVariable.unChainBackward();

                    model.tmpPredict = predictY;
                    processedBatches++;

                    System.out.println(String.format("    批次 %d 处理完成，损失: %.6f",
                            i + 1, lossValue));

                } catch (Exception e) {
                    System.err.println(String.format("  [线程-%d] 批次 %d 处理失败: %s",
                            threadId, i + 1, e.getMessage()));
                }
            }

            // 更新监控信息
            if (processedBatches > 0) {
                monitor.collectInfo(totalLoss / processedBatches);
            }
            monitor.printTrainInfo();

            long epochEndTime = System.currentTimeMillis();
            System.out.println(String.format("Epoch %d 完成，耗时: %d ms",
                    epoch, epochEndTime - epochStartTime));
        }

        // 跳过绘图以避免依赖问题
        // monitor.plot();
        System.out.println("简化版并行训练演示完成！");
    }

}