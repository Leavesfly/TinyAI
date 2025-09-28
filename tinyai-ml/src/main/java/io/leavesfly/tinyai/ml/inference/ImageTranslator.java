package io.leavesfly.tinyai.ml.inference;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import javax.imageio.ImageIO;

/**
 * 图像数据转换器
 * <p>
 * 用于图像数据与NdArray之间的相互转换。
 * 支持以下功能：
 * 1. 多种图像格式的输入支持
 * 2. 图像预处理功能（缩放、归一化等）
 * 3. 支持批量图像处理
 * 4. 灵活的输出格式化
 *
 * @author TinyDL
 * @version 1.0
 */
public class ImageTranslator implements Translator<Object, String> {
    
    /**
     * 目标图像宽度
     */
    private int targetWidth = 224;
    
    /**
     * 目标图像高度
     */
    private int targetHeight = 224;
    
    /**
     * 是否进行归一化（0-1范围）
     */
    private boolean normalize = true;
    
    /**
     * 归一化的均值（RGB通道）
     */
    private float[] meanValues = {0.485f, 0.456f, 0.406f};
    
    /**
     * 归一化的标准差（RGB通道）
     */
    private float[] stdValues = {0.229f, 0.224f, 0.225f};
    
    /**
     * 类别标签映射
     */
    private Map<Integer, String> classLabels = new HashMap<>();
    
    /**
     * 默认构造函数
     */
    public ImageTranslator() {
        initializeDefaultLabels();
    }
    
    /**
     * 构造函数
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @param normalize 是否归一化
     */
    public ImageTranslator(int targetWidth, int targetHeight, boolean normalize) {
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.normalize = normalize;
        initializeDefaultLabels();
    }
    
    /**
     * 初始化默认类别标签
     */
    private void initializeDefaultLabels() {
        // 这里可以添加一些默认的类别标签
        classLabels.put(0, "背景");
        classLabels.put(1, "物体");
    }
    
    /**
     * 设置类别标签映射
     * @param classLabels 类别标签映射
     */
    public void setClassLabels(Map<Integer, String> classLabels) {
        this.classLabels = classLabels;
    }
    
    /**
     * 设置归一化参数
     * @param meanValues 均值数组
     * @param stdValues 标准差数组
     */
    public void setNormalizationParams(float[] meanValues, float[] stdValues) {
        this.meanValues = meanValues.clone();
        this.stdValues = stdValues.clone();
    }

    @Override
    public NdArray input2NdArray(Object input) {
        try {
            if (input instanceof float[][]) {
                // 处理二维浮点数数组
                return handleFloatArray((float[][]) input);
            } else if (input instanceof BufferedImage) {
                // 处理BufferedImage
                return handleBufferedImage((BufferedImage) input);
            } else if (input instanceof String) {
                // 处理文件路径
                return handleImageFile((String) input);
            } else if (input instanceof File) {
                // 处理File对象
                return handleImageFile(((File) input).getAbsolutePath());
            } else if (input instanceof int[][][]) {
                // 处理RGB数组
                return handleRGBArray((int[][][]) input);
            } else {
                throw new IllegalArgumentException("不支持的输入类型: " + input.getClass().getName());
            }
        } catch (Exception e) {
            throw new RuntimeException("图像转换失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 处理二维浮点数数组
     */
    private NdArray handleFloatArray(float[][] data) {
        // 假设输入是灰度图像数据
        float[][][] processedData = preprocessGrayscaleImage(data);
        return NdArray.of(processedData);
    }
    
    /**
     * 处理BufferedImage
     */
    private NdArray handleBufferedImage(BufferedImage image) {
        // 缩放图像
        BufferedImage resizedImage = resizeImage(image, targetWidth, targetHeight);
        
        // 转换为RGB数组
        float[][][] rgbData = bufferedImageToRGBArray(resizedImage);
        
        // 归一化
        if (normalize) {
            normalizeRGBData(rgbData);
        }
        
        return NdArray.of(rgbData);
    }
    
    /**
     * 处理图像文件
     */
    private NdArray handleImageFile(String filePath) throws IOException {
        File file = new File(filePath);
        if (!file.exists()) {
            throw new IOException("图像文件不存在: " + filePath);
        }
        
        BufferedImage image = ImageIO.read(file);
        if (image == null) {
            throw new IOException("无法读取图像文件: " + filePath);
        }
        
        return handleBufferedImage(image);
    }
    
    /**
     * 处理RGB数组
     */
    private NdArray handleRGBArray(int[][][] rgbArray) {
        int height = rgbArray.length;
        int width = rgbArray[0].length;
        int channels = rgbArray[0][0].length;
        
        // 转换为float数组
        float[][][] floatData = new float[channels][height][width];
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    floatData[c][h][w] = rgbArray[h][w][c] / 255.0f;
                }
            }
        }
        
        // 缩放到目标尺寸
        if (height != targetHeight || width != targetWidth) {
            floatData = resizeRGBData(floatData, targetWidth, targetHeight);
        }
        
        // 归一化
        if (normalize) {
            normalizeRGBData(floatData);
        }
        
        return NdArray.of(floatData);
    }
    
    /**
     * 缩放图像
     */
    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        Image scaledImage = originalImage.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();
        g2d.drawImage(scaledImage, 0, 0, null);
        g2d.dispose();
        return resizedImage;
    }
    
    /**
     * 将BufferedImage转换为RGB数组
     */
    private float[][][] bufferedImageToRGBArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][][] rgbArray = new float[3][height][width]; // RGB三通道
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                rgbArray[0][y][x] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                rgbArray[1][y][x] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                rgbArray[2][y][x] = (rgb & 0xFF) / 255.0f;         // B
            }
        }
        
        return rgbArray;
    }
    
    /**
     * 预处理灰度图像
     */
    private float[][][] preprocessGrayscaleImage(float[][] grayData) {
        int height = grayData.length;
        int width = grayData[0].length;
        
        // 缩放到目标尺寸
        float[][] resizedData = resizeGrayscaleData(grayData, targetWidth, targetHeight);
        
        // 转换为3通道（复制灰度值到RGB三个通道）
        float[][][] rgbData = new float[3][targetHeight][targetWidth];
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < targetHeight; h++) {
                System.arraycopy(resizedData[h], 0, rgbData[c][h], 0, targetWidth);
            }
        }
        
        return rgbData;
    }
    
    /**
     * 缩放灰度图像数据（简单的最近邻插值）
     */
    private float[][] resizeGrayscaleData(float[][] data, int newWidth, int newHeight) {
        int oldHeight = data.length;
        int oldWidth = data[0].length;
        
        if (oldHeight == newHeight && oldWidth == newWidth) {
            return data;
        }
        
        float[][] resized = new float[newHeight][newWidth];
        float xRatio = (float) oldWidth / newWidth;
        float yRatio = (float) oldHeight / newHeight;
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                int oldX = (int) (x * xRatio);
                int oldY = (int) (y * yRatio);
                // 边界检查
                oldX = Math.min(oldX, oldWidth - 1);
                oldY = Math.min(oldY, oldHeight - 1);
                resized[y][x] = data[oldY][oldX];
            }
        }
        
        return resized;
    }
    
    /**
     * 缩放RGB数据
     */
    private float[][][] resizeRGBData(float[][][] data, int newWidth, int newHeight) {
        int channels = data.length;
        int oldHeight = data[0].length;
        int oldWidth = data[0][0].length;
        
        float[][][] resized = new float[channels][newHeight][newWidth];
        float xRatio = (float) oldWidth / newWidth;
        float yRatio = (float) oldHeight / newHeight;
        
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < newHeight; y++) {
                for (int x = 0; x < newWidth; x++) {
                    int oldX = (int) (x * xRatio);
                    int oldY = (int) (y * yRatio);
                    oldX = Math.min(oldX, oldWidth - 1);
                    oldY = Math.min(oldY, oldHeight - 1);
                    resized[c][y][x] = data[c][oldY][oldX];
                }
            }
        }
        
        return resized;
    }
    
    /**
     * 对RGB数据进行归一化
     */
    private void normalizeRGBData(float[][][] rgbData) {
        int channels = Math.min(rgbData.length, meanValues.length);
        int height = rgbData[0].length;
        int width = rgbData[0][0].length;
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    rgbData[c][h][w] = (rgbData[c][h][w] - meanValues[c]) / stdValues[c];
                }
            }
        }
    }

    @Override
    public String ndArray2Output(NdArray ndArray) {
        if (ndArray == null) {
            return "空结果";
        }
        
        try {
            Shape shape = ndArray.getShape();
            
            if (shape.getDimNum() == 1 || (shape.getDimNum() == 2 && shape.getDimension(0) == 1)) {
                // 单个预测结果
                return formatSinglePrediction(ndArray);
            } else if (shape.getDimNum() == 2) {
                // 批量预测结果
                return formatBatchPredictions(ndArray);
            } else {
                // 其他格式
                return formatGenericOutput(ndArray);
            }
        } catch (Exception e) {
            return "结果解析失败: " + e.getMessage();
        }
    }
    
    /**
     * 格式化单个预测结果
     */
    private String formatSinglePrediction(NdArray ndArray) {
        StringBuilder result = new StringBuilder();
        result.append("图像分类结果:\n");
        
        // 获取数组数据 - 由于NdArray接口中没有toFloatArray方法，使用其他方式
        float[] probabilities;
        try {
            // 尝试转换为一维数组
            probabilities = ndArray.flatten().getMatrix()[0];
        } catch (Exception e) {
            // 如果失败，尝试直接获取矩阵的第一行
            try {
                probabilities = ndArray.getMatrix()[0];
            } catch (Exception ex) {
                return "无法解析预测结果: " + ex.getMessage();
            }
        }
        
        // 找到最大概率的类别
        int maxIndex = 0;
        float maxProb = probabilities[0];
        
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        // 输出预测结果
        String className = classLabels.getOrDefault(maxIndex, "类别_" + maxIndex);
        result.append(String.format("预测类别: %s (ID: %d)\n", className, maxIndex));
        result.append(String.format("置信度: %.4f\n", maxProb));
        
        // 输出TOP-3结果
        if (probabilities.length > 1) {
            result.append("\nTOP-3结果:\n");
            
            // 创建索引数组并排序
            Integer[] indices = new Integer[probabilities.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }

            float[] finalProbabilities = probabilities;
            Arrays.sort(indices, (a, b) -> Float.compare(finalProbabilities[b], finalProbabilities[a]));
            
            int topK = Math.min(3, probabilities.length);
            for (int i = 0; i < topK; i++) {
                int idx = indices[i];
                String label = classLabels.getOrDefault(idx, "类别_" + idx);
                result.append(String.format("%d. %s: %.4f\n", i + 1, label, probabilities[idx]));
            }
        }
        
        return result.toString();
    }
    
    /**
     * 格式化批量预测结果
     */
    private String formatBatchPredictions(NdArray ndArray) {
        StringBuilder result = new StringBuilder();
        result.append("批量图像分类结果:\n");
        
        Shape shape = ndArray.getShape();
        int batchSize = shape.getDimension(0);
        int numClasses = shape.getDimension(1);
        
        float[][] allProbs = ndArray.getMatrix();
        
        for (int b = 0; b < batchSize; b++) {
            result.append(String.format("\n图像 %d:\n", b + 1));
            
            float[] probs = allProbs[b];
            
            // 找到最大概率
            int maxIndex = 0;
            for (int i = 1; i < probs.length; i++) {
                if (probs[i] > probs[maxIndex]) {
                    maxIndex = i;
                }
            }
            
            String className = classLabels.getOrDefault(maxIndex, "类别_" + maxIndex);
            result.append(String.format("  预测类别: %s, 置信度: %.4f\n", className, probs[maxIndex]));
        }
        
        return result.toString();
    }
    
    /**
     * 格式化通用输出
     */
    private String formatGenericOutput(NdArray ndArray) {
        StringBuilder result = new StringBuilder();
        result.append("图像处理结果:\n");
        result.append("输出形状: ").append(ndArray.getShape().toString()).append("\n");
        
        // 如果是小数组，显示数值
        int totalElements = ndArray.getShape().size();
        if (totalElements <= 20) {
            try {
                float[] data = ndArray.flatten().getMatrix()[0];
                result.append("数值: ").append(Arrays.toString(data)).append("\n");
            } catch (Exception e) {
                result.append("无法显示数值: ").append(e.getMessage()).append("\n");
            }
        } else {
            result.append("数组过大，仅显示统计信息:\n");
            try {
                float[] data = ndArray.flatten().getMatrix()[0];
                float min = Float.MAX_VALUE;
                float max = Float.MIN_VALUE;
                float sum = 0;
                
                for (float val : data) {
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                    sum += val;
                }
                
                result.append(String.format("  最小值: %.4f\n", min));
                result.append(String.format("  最大值: %.4f\n", max));
                result.append(String.format("  平均值: %.4f\n", sum / data.length));
            } catch (Exception e) {
                result.append("无法计算统计信息: ").append(e.getMessage()).append("\n");
            }
        }
        
        return result.toString();
    }
}