package io.leavesfly.tinyai.ml;



import io.leavesfly.tinyai.util.Utils;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * 模型训练监控器
 * 
 * 该类用于收集和可视化模型训练过程中的信息，包括：
 * 1. 训练损失值的收集和存储
 * 2. 验证集损失值的收集和存储
 * 3. 训练准确率的收集和存储
 * 4. 验证集准确率的收集和存储
 * 5. 训练时间的记录
 * 6. 训练信息的打印输出
 * 7. 训练过程的可视化展示
 * 8. 训练日志的保存
 * 
 * @author TinyDL
 * @version 1.0
 */
public class Monitor {

    private int index;

    List<Float> lossList;
    List<Float> valLossList;
    List<Float> accuracyList;
    List<Float> valAccuracyList;
    List<Long> timeList;
    
    private long epochStartTime;
    private String logFilePath;
    private boolean saveLogToFile;

    /**
     * 默认构造函数
     */
    public Monitor() {
        index = 0;
        this.lossList = new ArrayList<>();
        this.valLossList = new ArrayList<>();
        this.accuracyList = new ArrayList<>();
        this.valAccuracyList = new ArrayList<>();
        this.timeList = new ArrayList<>();
        this.saveLogToFile = false;
    }
    
    /**
     * 构造函数，支持日志文件保存
     * @param logFilePath 日志文件路径
     */
    public Monitor(String logFilePath) {
        this();
        this.logFilePath = logFilePath;
        this.saveLogToFile = true;
    }

    /**
     * 开始新的训练轮次
     * @param _index 轮次索引
     */
    public void startNewEpoch(int _index) {
        index = _index;
        epochStartTime = System.currentTimeMillis();
    }
    
    /**
     * 结束当前训练轮次并记录时间
     */
    public void endEpoch() {
        long epochEndTime = System.currentTimeMillis();
        timeList.add(epochEndTime - epochStartTime);
    }

    /**
     * 收集训练信息（损失值）
     * @param loss 当前批次的损失值
     */
    public void collectInfo(float loss) {
        lossList.add(loss);
    }
    
    /**
     * 收集验证集损失信息
     * @param valLoss 验证集损失值
     */
    public void collectValLoss(float valLoss) {
        valLossList.add(valLoss);
    }
    
    /**
     * 收集训练准确率信息
     * @param accuracy 训练准确率
     */
    public void collectAccuracy(float accuracy) {
        accuracyList.add(accuracy);
    }
    
    /**
     * 收集验证集准确率信息
     * @param valAccuracy 验证集准确率
     */
    public void collectValAccuracy(float valAccuracy) {
        valAccuracyList.add(valAccuracy);
    }

    /**
     * 打印训练信息
     */
    public void printTrainInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("epoch = ").append(index);
        
        if (lossList.size() > index) {
            sb.append(", loss: ").append(String.format("%.6f", lossList.get(index)));
        }
        
        if (accuracyList.size() > index) {
            sb.append(", acc: ").append(String.format("%.4f", accuracyList.get(index)));
        }
        
        if (timeList.size() > index) {
            sb.append(", time: ").append(timeList.get(index)).append("ms");
        }
        
        System.out.println(sb.toString());
        
        // 保存日志到文件
        if (saveLogToFile) {
            saveLogToFile(sb.toString());
        }
    }
    
    /**
     * 打印验证集信息
     */
    public void printValInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("epoch = ").append(index).append(" [validation]");
        
        if (valLossList.size() > index) {
            sb.append(", val_loss: ").append(String.format("%.6f", valLossList.get(index)));
        }
        
        if (valAccuracyList.size() > index) {
            sb.append(", val_acc: ").append(String.format("%.4f", valAccuracyList.get(index)));
        }
        
        System.out.println(sb.toString());
        
        // 保存日志到文件
        if (saveLogToFile) {
            saveLogToFile(sb.toString());
        }
    }
    
    /**
     * 保存日志到文件
     * @param logMessage 日志信息
     */
    private void saveLogToFile(String logMessage) {
        if (logFilePath == null || logFilePath.isEmpty()) {
            return;
        }
        
        try (FileWriter writer = new FileWriter(logFilePath, true)) {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            String timestamp = sdf.format(new Date());
            writer.write("[" + timestamp + "] " + logMessage + "\n");
        } catch (IOException e) {
            System.err.println("保存日志到文件时出错: " + e.getMessage());
        }
    }

    /**
     * 绘制训练过程图表
     */
    public void plot() {
        Plot plot = new Plot();
        int size = lossList.size();
        if (size > 0) {
            Float[] loss = lossList.toArray(new Float[0]);
            plot.line(Utils.toFloat(Utils.getSeq(size)), Utils.toFloat(loss), "train_loss");
        }
        
        if (valLossList.size() > 0) {
            Float[] valLoss = valLossList.toArray(new Float[0]);
            plot.line(Utils.toFloat(Utils.getSeq(valLoss.length)), Utils.toFloat(valLoss), "val_loss");
        }
        
        if (accuracyList.size() > 0) {
            Float[] acc = accuracyList.toArray(new Float[0]);
            plot.line(Utils.toFloat(Utils.getSeq(accuracyList.size())), Utils.toFloat(acc), "train_acc");
        }
        
        if (valAccuracyList.size() > 0) {
            Float[] valAcc = valAccuracyList.toArray(new Float[0]);
            plot.line(Utils.toFloat(Utils.getSeq(valAccuracyList.size())), Utils.toFloat(valAcc), "val_acc");
        }
        
        plot.show();
    }
    
    /**
     * 获取当前轮次索引
     * @return 当前轮次索引
     */
    public int getCurrentEpoch() {
        return index;
    }
    
    /**
     * 获取训练损失列表
     * @return 训练损失列表
     */
    public List<Float> getLossList() {
        return new ArrayList<>(lossList);
    }
    
    /**
     * 获取验证集损失列表
     * @return 验证集损失列表
     */
    public List<Float> getValLossList() {
        return new ArrayList<>(valLossList);
    }
    
    /**
     * 获取训练准确率列表
     * @return 训练准确率列表
     */
    public List<Float> getAccuracyList() {
        return new ArrayList<>(accuracyList);
    }
    
    /**
     * 获取验证集准确率列表
     * @return 验证集准确率列表
     */
    public List<Float> getValAccuracyList() {
        return new ArrayList<>(valAccuracyList);
    }
    
    /**
     * 获取训练时间列表
     * @return 训练时间列表
     */
    public List<Long> getTimeList() {
        return new ArrayList<>(timeList);
    }
    
    /**
     * 获取最佳训练损失值
     * @return 最佳训练损失值
     */
    public float getBestLoss() {
        if (lossList.isEmpty()) {
            return Float.MAX_VALUE;
        }
        return (float) lossList.stream().mapToDouble(Float::doubleValue).min().orElse(Float.MAX_VALUE);
    }
    
    /**
     * 获取最佳验证集损失值
     * @return 最佳验证集损失值
     */
    public float getBestValLoss() {
        if (valLossList.isEmpty()) {
            return Float.MAX_VALUE;
        }
        return (float) valLossList.stream().mapToDouble(Float::doubleValue).min().orElse(Float.MAX_VALUE);
    }
    
    /**
     * 获取最佳训练准确率
     * @return 最佳训练准确率
     */
    public float getBestAccuracy() {
        if (accuracyList.isEmpty()) {
            return 0.0f;
        }
        return (float) accuracyList.stream().mapToDouble(Float::doubleValue).max().orElse(0.0);
    }
    
    /**
     * 获取最佳验证集准确率
     * @return 最佳验证集准确率
     */
    public float getBestValAccuracy() {
        if (valAccuracyList.isEmpty()) {
            return 0.0f;
        }
        return (float) valAccuracyList.stream().mapToDouble(Float::doubleValue).max().orElse(0.0);
    }
}