package io.leavesfly.tinyai.mlearning;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.Parameter;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * 模型序列化器 - 提供完整的模型保存和加载功能
 * <p>
 * 该类提供了多种模型序列化和反序列化的方法，支持：
 * 1. 完整模型保存（包含架构和参数）
 * 2. 仅参数保存
 * 3. 压缩保存
 * 4. 模型检查点
 * 5. 模型验证和比较
 *
 * @author TinyDL
 * @version 1.0
 */
public class ModelSerializer {

    public static final String MODEL_INFO_SUFFIX = ".info";
    public static final String MODEL_PARAMS_SUFFIX = ".params";
    public static final String MODEL_COMPLETE_SUFFIX = ".model";
    public static final String MODEL_CHECKPOINT_SUFFIX = ".ckpt";

    /**
     * 保存完整模型（架构 + 参数）
     *
     * @param model    要保存的模型
     * @param filePath 保存路径
     * @param compress 是否压缩
     */
    public static void saveModel(Model model, String filePath, boolean compress) {
        try {
            File file = new File(filePath);
            createDirectoryIfNotExists(file.getParentFile());

            if (compress) {
                try (FileOutputStream fos = new FileOutputStream(file);
                     GZIPOutputStream gzos = new GZIPOutputStream(fos);
                     ObjectOutputStream oos = new ObjectOutputStream(gzos)) {
                    oos.writeObject(model);
                }
            } else {
                try (FileOutputStream fos = new FileOutputStream(file);
                     ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                    oos.writeObject(model);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save model: " + e.getMessage(), e);
        }
    }

    /**
     * 保存完整模型（默认不压缩）
     *
     * @param model    要保存的模型
     * @param filePath 保存路径
     */
    public static void saveModel(Model model, String filePath) {
        saveModel(model, filePath, false);
    }

    /**
     * 加载完整模型
     *
     * @param filePath   模型文件路径
     * @param compressed 是否为压缩文件
     * @return 加载的模型
     */
    public static Model loadModel(String filePath, boolean compressed) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                throw new RuntimeException("Model file does not exist: " + filePath);
            }

            if (compressed) {
                try (FileInputStream fis = new FileInputStream(file);
                     GZIPInputStream gzis = new GZIPInputStream(fis);
                     ObjectInputStream ois = new ObjectInputStream(gzis)) {
                    return (Model) ois.readObject();
                }
            } else {
                try (FileInputStream fis = new FileInputStream(file);
                     ObjectInputStream ois = new ObjectInputStream(fis)) {
                    return (Model) ois.readObject();
                }
            }
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Failed to load model: " + e.getMessage(), e);
        }
    }

    /**
     * 加载完整模型（自动检测是否压缩）
     *
     * @param filePath 模型文件路径
     * @return 加载的模型
     */
    public static Model loadModel(String filePath) {
        // 首先尝试非压缩加载
        try {
            return loadModel(filePath, false);
        } catch (Exception e) {
            // 如果失败，尝试压缩格式加载
            try {
                return loadModel(filePath, true);
            } catch (Exception e2) {
                throw new RuntimeException("Failed to load model, tried both compressed and uncompressed formats", e2);
            }
        }
    }

    /**
     * 仅保存模型参数
     *
     * @param model    模型
     * @param filePath 保存路径
     */
    public static void saveParameters(Model model, String filePath) {
        try {
            File file = new File(filePath);
            createDirectoryIfNotExists(file.getParentFile());

            Map<String, Parameter> params = model.getAllParams();

            try (FileOutputStream fos = new FileOutputStream(file);
                 ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                oos.writeObject(params);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save parameters: " + e.getMessage(), e);
        }
    }

    /**
     * 加载模型参数到现有模型中
     *
     * @param model    目标模型
     * @param filePath 参数文件路径
     *                                                                 todo
     */
    @SuppressWarnings("unchecked")
    public static void loadParameters(Model model, String filePath) {
        //todo
    }

    /**
     * 保存训练检查点（包含模型状态和训练信息）
     *
     * @param model    模型
     * @param epoch    当前训练轮次
     * @param loss     当前损失
     * @param filePath 保存路径
     */
    public static void saveCheckpoint(Model model, int epoch, double loss, String filePath) {
        try {
            File file = new File(filePath);
            createDirectoryIfNotExists(file.getParentFile());

            Map<String, Object> checkpoint = new HashMap<>();
            checkpoint.put("model", model);
            checkpoint.put("epoch", epoch);
            checkpoint.put("loss", loss);
            checkpoint.put("timestamp", System.currentTimeMillis());
            checkpoint.put("version", "TinyDL-0.01");

            try (FileOutputStream fos = new FileOutputStream(file);
                 ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                oos.writeObject(checkpoint);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save checkpoint: " + e.getMessage(), e);
        }
    }

    /**
     * 加载训练检查点
     *
     * @param filePath 检查点文件路径
     * @return 检查点信息（包含模型、轮次、损失等）
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadCheckpoint(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists()) {
                throw new RuntimeException("Checkpoint file does not exist: " + filePath);
            }

            try (FileInputStream fis = new FileInputStream(file);
                 ObjectInputStream ois = new ObjectInputStream(fis)) {
                return (Map<String, Object>) ois.readObject();
            }
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Failed to load checkpoint: " + e.getMessage(), e);
        }
    }

    /**
     * 从检查点恢复训练
     *
     * @param filePath 检查点文件路径
     * @return 恢复的模型
     */
    public static Model resumeFromCheckpoint(String filePath) {
        Map<String, Object> checkpoint = loadCheckpoint(filePath);
        Model model = (Model) checkpoint.get("model");

        System.out.println("Resumed from checkpoint:");
        System.out.println("  Epoch: " + checkpoint.get("epoch"));
        System.out.println("  Loss: " + checkpoint.get("loss"));
        System.out.println("  Timestamp: " + new java.util.Date((Long) checkpoint.get("timestamp")));

        return model;
    }

    /**
     * 获取模型文件大小
     *
     * @param filePath 文件路径
     * @return 文件大小（字节）
     */
    public static long getModelSize(String filePath) {
        File file = new File(filePath);
        return file.exists() ? file.length() : -1;
    }

    /**
     * 验证模型文件是否有效
     *
     * @param filePath 文件路径
     * @return 是否有效
     */
    public static boolean validateModelFile(String filePath) {
        try {
            loadModel(filePath);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 创建目录（如果不存在）
     *
     * @param directory 目录
     */
    private static void createDirectoryIfNotExists(File directory) {
        if (directory != null && !directory.exists()) {
            directory.mkdirs();
        }
    }

    /**
     * 比较两个模型的参数
     *
     * @param model1 模型1
     * @param model2 模型2
     * @return 参数是否相同
     * todo
     */
    public static boolean compareModelParameters(Model model1, Model model2) {
        //todo
        return false;
    }
}