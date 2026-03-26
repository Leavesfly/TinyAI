package io.leavesfly.tinyai.deepseek.base.utils;

import io.leavesfly.tinyai.ml.model.Model;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * DeepSeek 系列检查点管理器
 * 
 * 提供统一的检查点管理功能：
 * - 检查点保存（支持多种命名策略）
 * - 检查点加载
 * - 元数据记录（训练状态、配置信息）
 * - 自动清理旧检查点
 * 
 * @author leavesfly
 * @version 1.0
 */
public class CheckpointManager {
    
    private final String checkpointDir;
    private final int maxCheckpoints;  // 保留的最大检查点数（0表示不限制）
    
    /**
     * 构造函数
     * 
     * @param checkpointDir 检查点目录
     */
    public CheckpointManager(String checkpointDir) {
        this(checkpointDir, 0);
    }
    
    /**
     * 构造函数（带自动清理）
     * 
     * @param checkpointDir 检查点目录
     * @param maxCheckpoints 最大保留检查点数（0表示不限制）
     */
    public CheckpointManager(String checkpointDir, int maxCheckpoints) {
        this.checkpointDir = checkpointDir;
        this.maxCheckpoints = maxCheckpoints;
        createDirectory();
    }
    
    /**
     * 创建检查点目录
     */
    private void createDirectory() {
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    // ========== 检查点保存 ==========
    
    /**
     * 保存检查点（轮次命名）
     * 
     * @param model 模型
     * @param epoch 当前轮次
     * @param loss 当前损失
     * @param metadata 额外元数据
     * @return 检查点文件路径
     */
    public String saveEpochCheckpoint(Model model, int epoch, double loss, 
                                     Map<String, String> metadata) {
        String filename = String.format("%s_epoch_%d.model", 
                                       model.getName(), epoch);
        return saveCheckpoint(model, filename, epoch, loss, metadata);
    }
    
    /**
     * 保存最佳检查点
     */
    public String saveBestCheckpoint(Model model, int epoch, double loss, 
                                    Map<String, String> metadata) {
        String filename = String.format("%s_best.model", model.getName());
        return saveCheckpoint(model, filename, epoch, loss, metadata);
    }
    
    /**
     * 保存最终检查点
     */
    public String saveFinalCheckpoint(Model model, int epoch, double loss, 
                                     Map<String, String> metadata) {
        String filename = String.format("%s_final.model", model.getName());
        return saveCheckpoint(model, filename, epoch, loss, metadata);
    }
    
    /**
     * 保存检查点（时间戳命名）
     */
    public String saveTimestampCheckpoint(Model model, int epoch, double loss, 
                                         Map<String, String> metadata) {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = String.format("%s_%s.model", model.getName(), timestamp);
        return saveCheckpoint(model, filename, epoch, loss, metadata);
    }
    
    /**
     * 核心保存方法
     */
    private String saveCheckpoint(Model model, String filename, int epoch, 
                                 double loss, Map<String, String> metadata) {
        try {
            String filepath = Paths.get(checkpointDir, filename).toString();
            
            // 保存模型
            model.saveModel(filepath);
            
            // 保存元数据
            saveMetadata(filename, epoch, loss, metadata);
            
            // 清理旧检查点
            if (maxCheckpoints > 0) {
                cleanupOldCheckpoints(model.getName());
            }
            
            return filepath;
        } catch (Exception e) {
            System.err.println("❌ 保存检查点失败: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * 保存元数据文件
     */
    private void saveMetadata(String modelFilename, int epoch, double loss, 
                             Map<String, String> metadata) {
        String metaFilename = modelFilename.replace(".model", ".meta");
        String filepath = Paths.get(checkpointDir, metaFilename).toString();
        
        try (FileWriter writer = new FileWriter(filepath)) {
            writer.write("# DeepSeek 检查点元数据\n");
            writer.write("timestamp=" + new Date() + "\n");
            writer.write("epoch=" + epoch + "\n");
            writer.write("loss=" + String.format("%.6f", loss) + "\n");
            
            if (metadata != null) {
                for (Map.Entry<String, String> entry : metadata.entrySet()) {
                    writer.write(entry.getKey() + "=" + entry.getValue() + "\n");
                }
            }
        } catch (IOException e) {
            System.err.println("⚠️ 保存元数据失败: " + e.getMessage());
        }
    }
    
    // ========== 检查点加载 ==========
    
    /**
     * 加载指定检查点
     */
    public boolean loadCheckpoint(Model model, String filename) {
        String filepath = Paths.get(checkpointDir, filename).toString();
        File file = new File(filepath);
        
        if (!file.exists()) {
            System.err.println("❌ 检查点不存在: " + filepath);
            return false;
        }
        
        try {
            model.loadModel(filepath);
            System.out.println("✅ 检查点加载成功: " + filepath);
            return true;
        } catch (Exception e) {
            System.err.println("❌ 加载检查点失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 加载最佳检查点
     */
    public boolean loadBestCheckpoint(Model model) {
        String filename = String.format("%s_best.model", model.getName());
        return loadCheckpoint(model, filename);
    }
    
    /**
     * 加载最终检查点
     */
    public boolean loadFinalCheckpoint(Model model) {
        String filename = String.format("%s_final.model", model.getName());
        return loadCheckpoint(model, filename);
    }
    
    /**
     * 加载最新检查点
     */
    public boolean loadLatestCheckpoint(Model model) {
        File dir = new File(checkpointDir);
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith(model.getName()) && name.endsWith(".model"));
        
        if (files == null || files.length == 0) {
            System.err.println("❌ 未找到检查点");
            return false;
        }
        
        // 按修改时间排序，获取最新的
        Arrays.sort(files, Comparator.comparingLong(File::lastModified).reversed());
        
        return loadCheckpoint(model, files[0].getName());
    }
    
    // ========== 检查点管理 ==========
    
    /**
     * 清理旧检查点（保留最新的N个）
     */
    private void cleanupOldCheckpoints(String modelName) {
        File dir = new File(checkpointDir);
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith(modelName) && name.endsWith(".model") &&
            !name.endsWith("_best.model") && !name.endsWith("_final.model"));
        
        if (files == null || files.length <= maxCheckpoints) {
            return;
        }
        
        // 按修改时间排序
        Arrays.sort(files, Comparator.comparingLong(File::lastModified).reversed());
        
        // 删除超出数量的检查点
        for (int i = maxCheckpoints; i < files.length; i++) {
            String filename = files[i].getName();
            boolean deleted = files[i].delete();
            if (!deleted) {
                System.err.println("⚠️ 删除检查点失败: " + filename);
            }
            
            // 同时删除元数据文件
            String metaFilename = filename.replace(".model", ".meta");
            File metaFile = new File(dir, metaFilename);
            if (metaFile.exists() && !metaFile.delete()) {
                System.err.println("⚠️ 删除元数据文件失败: " + metaFilename);
            }
            
            if (deleted) {
                System.out.println("🗑️ 已清理旧检查点: " + filename);
            }
        }
    }
    
    /**
     * 列出所有检查点
     */
    public List<String> listCheckpoints(String modelName) {
        File dir = new File(checkpointDir);
        File[] files = dir.listFiles((d, name) -> 
            name.startsWith(modelName) && name.endsWith(".model"));
        
        if (files == null) {
            return new ArrayList<>();
        }
        
        List<String> checkpoints = new ArrayList<>();
        for (File file : files) {
            checkpoints.add(file.getName());
        }
        
        return checkpoints;
    }
    
    /**
     * 检查点是否存在
     */
    public boolean checkpointExists(String filename) {
        String filepath = Paths.get(checkpointDir, filename).toString();
        return new File(filepath).exists();
    }
    
    /**
     * 获取检查点目录
     */
    public String getCheckpointDir() {
        return checkpointDir;
    }
}
