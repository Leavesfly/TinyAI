package io.leavesfly.tinyai.ml.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.visual.Monitor;
import io.leavesfly.tinyai.ml.callback.TrainingCallback;
import io.leavesfly.tinyai.ml.dataset.Batch;
import io.leavesfly.tinyai.ml.dataset.DataSet;
import io.leavesfly.tinyai.ml.evaluator.Evaluator;
import io.leavesfly.tinyai.ml.loss.Loss;
import io.leavesfly.tinyai.ml.model.Model;
import io.leavesfly.tinyai.ml.optimize.Optimizer;
import io.leavesfly.tinyai.ml.training.parallel.GradientAggregator;
import io.leavesfly.tinyai.ml.training.parallel.ParallelBatchProcessor;
import io.leavesfly.tinyai.ml.training.parallel.ParallelTrainingUtils;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * 模型训练器。
 *
 * 该类是 TinyDL 框架中模型训练的核心组件，提供完整的训练流程管理，
 * 支持单线程和并行训练两种模式。
 *
 * 主要功能：
 * 1. 训练流程管理：控制训练的轮次、批次处理等
 * 2. 单线程训练：传统的顺序训练模式
 * 3. 并行训练：支持多线程并行处理批次数据
 * 4. 训练监控：与 Monitor 配合收集训练过程信息
 * 5. 模型评估：与 Evaluator 配合进行模型性能评估
 *
 * 使用示例：
 * 快速开始 - 最小配置：
 *   Trainer.builder()
 *       .model(model).dataSet(dataSet).loss(loss).optimizer(optimizer)
 *       .epochs(100).build().train();
 *
 * 完整配置 - 自定义所有参数：
 *   Trainer.builder()
 *       .model(model).dataSet(dataSet).loss(loss).optimizer(optimizer)
 *       .epochs(100).monitor(new Monitor("log.txt")).evaluator(evaluator)
 *       .enableParallel(true, 4).shuffle(true).validationInterval(5)
 *       .earlyStopping(patience, minDelta).gradientClipping(maxNorm)
 *       .build().train();
 *
 * @author TinyDL
 * @version 2.0
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
    
    // 验证集相关配置
    private DataSet validationSet;
    private Evaluator validationEvaluator;
    private int validationInterval = 1; // 每N个epoch评估一次
    
    // 早停机制
    private EarlyStopping earlyStopping;
    
    // 梯度裁剪
    private GradientClipper gradientClipper;
    
    // 回调机制
    private List<TrainingCallback> callbacks = new ArrayList<>();
    
    // 是否自动打乱数据
    private boolean shuffle = true;

    /**
     * 私有构造函数 - 强制使用Builder
     */
    private Trainer() {
    }

    /**
     * 创建Trainer构建器
     * 
     * @return Trainer构建器实例
     */
    public static Builder builder() {
        return new Builder();
    }

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
     * 主训练方法 - 使用默认shuffle设置
     */
    public void train() {
        train(shuffle);
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
        // 通知训练开始
        notifyTrainingStart();
        
        DataSet trainDataSet = dataSet.getTrainDataSet();
        if (shuffleData) {
            trainDataSet.shuffle();
        }

        float finalLoss = 0f;
        for (int i = 0; i < maxEpoch; i++) {
            // 检查是否应该停止
            if (shouldStopTraining()) {
                System.out.println("训练被回调提前停止");
                break;
            }

            model.resetState();
            monitor.startNewEpoch(i);
            notifyEpochStart(i);

            List<Batch> batches = trainDataSet.getBatches();
            float lossSum = 0f;

            for (int batchIndex = 0; batchIndex < batches.size(); batchIndex++) {
                Batch batch = batches.get(batchIndex);
                Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
                Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);

                Variable predictY = model.forward(variableX);
                Variable lossVariable = loss.loss(variableY, predictY);
                lossVariable.setName("loss");

                model.clearGrads();
                float batchLoss = lossVariable.getValue().getNumber().floatValue();
                lossSum += batchLoss;

                lossVariable.backward();
                
                // 梯度裁剪
                if (gradientClipper != null) {
                    gradientClipper.clipGradients(model);
                }

                optimizer.update();
                lossVariable.unChainBackward();

                model.tmpPredict = predictY;
                
                // 通知批次结束
                notifyBatchEnd(i, batchIndex, batchLoss);
            }
            
            float avgLoss = lossSum / batches.size();
            finalLoss = avgLoss;
            monitor.collectInfo(avgLoss);
            monitor.endEpoch();
            monitor.printTrainInfo();
            
            // 验证集评估
            if (validationSet != null && (i + 1) % validationInterval == 0) {
                validate(i);
            }
            
            // 早停检查
            if (earlyStopping != null) {
                float checkValue = validationSet != null ? 
                    monitor.getValLossList().isEmpty() ? avgLoss : 
                    monitor.getValLossList().get(monitor.getValLossList().size() - 1) : avgLoss;
                
                if (earlyStopping.shouldStop(checkValue, model)) {
                    System.out.println("早停机制触发，训练提前结束");
                    break;
                }
            }
            
            // 通知epoch结束
            notifyEpochEnd(i, avgLoss, null);
        }
        
        // 通知训练结束
        notifyTrainingEnd(maxEpoch, finalLoss);
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
     * 设置验证集
     * 
     * @param validationSet 验证集
     * @param validationEvaluator 验证集评估器
     * @param interval 验证间隔（每N个epoch评估一次）
     */
    public void setValidationSet(DataSet validationSet, Evaluator validationEvaluator, int interval) {
        this.validationSet = validationSet;
        this.validationEvaluator = validationEvaluator;
        this.validationInterval = interval > 0 ? interval : 1;
    }
    
    /**
     * 设置早停机制
     * 
     * @param earlyStopping 早停对象
     */
    public void setEarlyStopping(EarlyStopping earlyStopping) {
        this.earlyStopping = earlyStopping;
    }
    
    /**
     * 设置梯度裁剪
     * 
     * @param gradientClipper 梯度裁剪器
     */
    public void setGradientClipper(GradientClipper gradientClipper) {
        this.gradientClipper = gradientClipper;
    }
    
    /**
     * 添加训练回调
     * 
     * @param callback 回调对象
     */
    public void addCallback(TrainingCallback callback) {
        if (callback != null) {
            callbacks.add(callback);
        }
    }
    
    /**
     * 移除训练回调
     * 
     * @param callback 回调对象
     */
    public void removeCallback(TrainingCallback callback) {
        callbacks.remove(callback);
    }
    
    /**
     * 执行验证集评估
     * 
     * @param epoch 当前轮次
     */
    private void validate(int epoch) {
        if (validationSet == null || validationEvaluator == null) {
            return;
        }
        
        // 通知回调验证开始
        for (TrainingCallback callback : callbacks) {
            callback.onValidationStart(epoch);
        }
        
        // 设置模型为评估模式
        model.getModule().eval();
        
        // 计算验证损失
        float valLoss = evaluateLoss(validationSet);
        
        // 执行验证评估
        validationEvaluator.evaluate();
        
        // 获取验证准确率（如果可用）
        // 注意：这里需要根据实际Evaluator实现获取准确率
        // 如果Evaluator提供了获取准确率的方法，可以在这里调用
        Float valAccuracy = null;
        
        // 收集验证信息到监控器
        monitor.collectValLoss(valLoss);
        // 如果获取到验证准确率，可以调用 monitor.collectValAccuracy(valAccuracy);
        
        // 通知回调验证结束
        for (TrainingCallback callback : callbacks) {
            callback.onValidationEnd(epoch, valLoss, valAccuracy);
        }
        
        // 恢复训练模式
        model.getModule().train();
    }
    
    /**
     * 评估数据集上的损失
     * 
     * @param dataSet 数据集
     * @return 平均损失
     */
    private float evaluateLoss(DataSet dataSet) {
        List<Batch> batches = dataSet.getBatches();
        float lossSum = 0f;
        
        for (Batch batch : batches) {
            Variable variableX = batch.toVariableX().setName("x").setRequireGrad(false);
            Variable variableY = batch.toVariableY().setName("y").setRequireGrad(false);
            
            Variable predictY = model.forward(variableX);
            Variable lossVariable = loss.loss(variableY, predictY);
            
            lossSum += lossVariable.getValue().getNumber().floatValue();
        }
        
        return lossSum / batches.size();
    }
    
    /**
     * 通知回调训练开始
     */
    private void notifyTrainingStart() {
        for (TrainingCallback callback : callbacks) {
            callback.onTrainingStart();
        }
    }
    
    /**
     * 通知回调训练结束
     */
    private void notifyTrainingEnd(int epoch, float finalLoss) {
        for (TrainingCallback callback : callbacks) {
            callback.onTrainingEnd(epoch, finalLoss);
        }
    }
    
    /**
     * 通知回调epoch开始
     */
    private void notifyEpochStart(int epoch) {
        for (TrainingCallback callback : callbacks) {
            callback.onEpochStart(epoch);
        }
    }
    
    /**
     * 通知回调epoch结束
     */
    private void notifyEpochEnd(int epoch, float loss, Float accuracy) {
        for (TrainingCallback callback : callbacks) {
            callback.onEpochEnd(epoch, loss, accuracy);
        }
    }
    
    /**
     * 通知回调批次结束
     */
    private void notifyBatchEnd(int epoch, int batchIndex, float batchLoss) {
        for (TrainingCallback callback : callbacks) {
            callback.onBatchEnd(epoch, batchIndex, batchLoss);
        }
    }
    
    /**
     * 检查是否应该停止训练
     */
    private boolean shouldStopTraining() {
        // 检查回调
        for (TrainingCallback callback : callbacks) {
            if (callback.shouldStop()) {
                return true;
            }
        }
        return false;
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

    /**
     * Trainer 构建器，提供流畅的 API 来配置和创建 Trainer 实例。
     */
    public static class Builder {
        private Model model;
        private DataSet dataSet;
        private Loss loss;
        private Optimizer optimizer;
        private int epochs = 100;
        private Monitor monitor;
        private Evaluator evaluator;
        private boolean enableParallel = false;
        private int threadCount = 0;
        private boolean shuffle = true;
        private int validationInterval = 1;
        private Integer earlyStoppingPatience;
        private Float earlyStoppingMinDelta;
        private Float gradientClipMaxNorm;
        private String monitorLogFile;

        /**
         * 设置模型 (必需)
         */
        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        /**
         * 设置数据集 (必需)
         */
        public Builder dataSet(DataSet dataSet) {
            this.dataSet = dataSet;
            return this;
        }

        /**
         * 设置损失函数 (必需)
         */
        public Builder loss(Loss loss) {
            this.loss = loss;
            return this;
        }

        /**
         * 设置优化器 (必需)
         */
        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        /**
         * 设置训练轮次 (默认100)
         */
        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        /**
         * 设置Monitor
         */
        public Builder monitor(Monitor monitor) {
            this.monitor = monitor;
            return this;
        }

        /**
         * 设置Monitor日志文件路径
         */
        public Builder monitorLogFile(String logFile) {
            this.monitorLogFile = logFile;
            return this;
        }

        /**
         * 设置评估器
         */
        public Builder evaluator(Evaluator evaluator) {
            this.evaluator = evaluator;
            return this;
        }

        /**
         * 启用并行训练
         * 
         * @param enable 是否启用
         * @param threadCount 线程数(0表示自动)
         */
        public Builder enableParallel(boolean enable, int threadCount) {
            this.enableParallel = enable;
            this.threadCount = threadCount;
            return this;
        }

        /**
         * 启用并行训练(自动线程数)
         */
        public Builder enableParallel(boolean enable) {
            return enableParallel(enable, 0);
        }

        /**
         * 设置是否打乱数据 (默认true)
         */
        public Builder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }

        /**
         * 设置验证间隔
         */
        public Builder validationInterval(int interval) {
            this.validationInterval = interval;
            return this;
        }

        /**
         * 启用早停机制
         * 
         * @param patience 容忍轮次
         * @param minDelta 最小改进量
         */
        public Builder earlyStopping(int patience, float minDelta) {
            this.earlyStoppingPatience = patience;
            this.earlyStoppingMinDelta = minDelta;
            return this;
        }

        /**
         * 启用梯度裁剪
         * 
         * @param maxNorm 最大梯度范数
         */
        public Builder gradientClipping(float maxNorm) {
            this.gradientClipMaxNorm = maxNorm;
            return this;
        }

        /**
         * 构建Trainer实例
         */
        public Trainer build() {
            // 验证必需参数
            if (model == null) {
                throw new IllegalStateException("Model is required");
            }
            if (dataSet == null) {
                throw new IllegalStateException("DataSet is required");
            }
            if (loss == null) {
                throw new IllegalStateException("Loss is required");
            }
            if (optimizer == null) {
                throw new IllegalStateException("Optimizer is required");
            }

            Trainer trainer = new Trainer();
            trainer.model = this.model;
            trainer.dataSet = this.dataSet;
            trainer.loss = this.loss;
            trainer.optimizer = this.optimizer;
            trainer.maxEpoch = this.epochs;
            trainer.shuffle = this.shuffle;
            trainer.validationInterval = this.validationInterval;
            trainer.enableParallelTraining = this.enableParallel;
            trainer.parallelThreadCount = this.threadCount > 0 ? this.threadCount :
                    ParallelTrainingUtils.getRecommendedThreadCount(4);

            // 创建默认Monitor或使用提供的
            if (this.monitor != null) {
                trainer.monitor = this.monitor;
            } else if (this.monitorLogFile != null) {
                trainer.monitor = new Monitor(this.monitorLogFile);
            } else {
                trainer.monitor = new Monitor();
            }

            // 创建默认Evaluator或使用提供的
            if (this.evaluator != null) {
                trainer.evaluator = this.evaluator;
            } else {
                // 根据loss类型创建默认evaluator
                trainer.evaluator = createDefaultEvaluator(loss, model, dataSet);
            }

            // 配置早停
            if (earlyStoppingPatience != null) {
                trainer.earlyStopping = new EarlyStopping(
                        earlyStoppingPatience,
                        earlyStoppingMinDelta != null ? earlyStoppingMinDelta : 0.0001f,
                        EarlyStopping.MonitorMode.MIN,  // 默认监控loss最小值
                        true  // 恢复最佳权重
                );
            }

            // 配置梯度裁剪
            if (gradientClipMaxNorm != null) {
                trainer.gradientClipper = new GradientClipper(
                        GradientClipper.ClipType.NORM,  // 默认使用L2范数裁剪
                        gradientClipMaxNorm
                );
            }

            // 初始化训练器
            trainer.dataSet.prepare();

            // 检查并行训练支持
            if (trainer.enableParallelTraining && 
                !ParallelTrainingUtils.isModelParallelizable(trainer.model)) {
                System.err.println("警告: 模型不支持并行训练，将回退到单线程模式");
                trainer.enableParallelTraining = false;
            }

            // 初始化线程池
            if (trainer.enableParallelTraining) {
                DataSet trainDataSet = trainer.dataSet.getTrainDataSet();
                if (trainDataSet != null) {
                    List<Batch> batches = trainDataSet.getBatches();
                    trainer.parallelThreadCount = Math.min(
                            trainer.parallelThreadCount, batches.size());
                }
                trainer.executorService = Executors.newFixedThreadPool(
                        trainer.parallelThreadCount);
                System.out.println("并行训练已启用，线程数: " + trainer.parallelThreadCount);
            }

            return trainer;
        }

        /**
         * 创建默认评估器
         */
        private Evaluator createDefaultEvaluator(Loss loss, Model model, DataSet dataSet) {
            // 简单的默认实现 - 可以根据loss类型选择合适的evaluator
            return new Evaluator() {
                @Override
                public void evaluate() {
                    // 默认不做评估
                    System.out.println("[提示] 未配置评估器，跳过评估");
                }
            };
        }
    }

}