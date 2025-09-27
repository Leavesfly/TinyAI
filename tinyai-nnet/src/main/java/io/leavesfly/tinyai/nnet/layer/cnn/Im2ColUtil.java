package io.leavesfly.tinyai.nnet.layer.cnn;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 优化的Im2Col工具类
 * 当处理四维数组时，我们通常处理的是多个样本（例如，一批图像）其中每个样本可能包含多个通道（例如，RGB图像有三个通道）
 * 增加了缓存机制和性能优化
 */
public class Im2ColUtil {
    
    // 缓存机制：缓存已经分配的数组以避免重复分配
    private static final Map<String, float[][]> outputCache = new ConcurrentHashMap<>();
    private static final int MAX_CACHE_SIZE = 10;  // 最大缓存数量
    
    /**
     * 生成缓存键
     */
    private static String generateCacheKey(int numSamples, int outHeight, int outWidth, int channels, int filterH, int filterW) {
        return String.format("%d_%d_%d_%d_%d_%d", numSamples, outHeight, outWidth, channels, filterH, filterW);
    }
    
    /**
     * 获取或创建输出数组
     */
    private static float[][] getOrCreateOutput(String cacheKey, int rows, int cols) {
        float[][] output = outputCache.get(cacheKey);
        if (output == null || output.length != rows || output[0].length != cols) {
            output = new float[rows][cols];
            if (outputCache.size() >= MAX_CACHE_SIZE) {
                // 简单的LRU策略：清空缓存
                outputCache.clear();
            }
            outputCache.put(cacheKey, output);
        } else {
            // 清零复用的数组
            for (int i = 0; i < rows; i++) {
                java.util.Arrays.fill(output[i], 0.0f);
            }
        }
        return output;
    }

    /**
     * 优化版本：对四维输入数组执行 im2col 操作。
     * 增加了缓存机制和性能优化
     *
     * @param input   预期形状为 [numSamples][channels][height][width] 的四维数组。
     * @param filterH 滤波器高度。
     * @param filterW 滤波器宽度。
     * @param stride  步长。
     * @param pad     填充。
     * @return 展开后的二维数组。
     */
    public static float[][] im2col(float[][][][] input, int filterH, int filterW, int stride, int pad) {
        int numSamples = input.length; // 样本数量（batch size）
        int channels = input[0].length; // 颜色通道数量
        int height = input[0][0].length; // 输入图像的高度
        int width = input[0][0][0].length; // 输入图像的宽度

        // 计算输出的高度和宽度
        int outHeight = (height + 2 * pad - filterH) / stride + 1;
        int outWidth = (width + 2 * pad - filterW) / stride + 1;

        // 使用缓存机制获取输出数组
        String cacheKey = generateCacheKey(numSamples, outHeight, outWidth, channels, filterH, filterW);
        int outputRows = numSamples * outHeight * outWidth;
        int outputCols = channels * filterH * filterW;
        float[][] output = getOrCreateOutput(cacheKey, outputRows, outputCols);

        // 优化的循环顺序：按照内存访问模式优化
        int outputRowIndex = 0;
        for (int n = 0; n < numSamples; n++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int colIndex = 0;
                    
                    // 对于每个颜色通道
                    for (int c = 0; c < channels; c++) {
                        // 对于卷积核的每个位置
                        for (int fh = 0; fh < filterH; fh++) {
                            int imRow = h * stride + fh - pad;
                            for (int fw = 0; fw < filterW; fw++) {
                                int imCol = w * stride + fw - pad;

                                // 判断是否在输入图像的边界之内，否则填充0
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    output[outputRowIndex][colIndex] = input[n][c][imRow][imCol];
                                } else {
                                    output[outputRowIndex][colIndex] = 0.0f; // 应用填充
                                }
                                colIndex++;
                            }
                        }
                    }
                    outputRowIndex++;
                }
            }
        }
        return output;
    }
    
    /**
     * 高效版本：使用一维数组存储的更快版本
     * 适用于大批量数据处理
     */
    public static float[][] im2colFast(float[][][][] input, int filterH, int filterW, int stride, int pad) {
        int numSamples = input.length;
        int channels = input[0].length;
        int height = input[0][0].length;
        int width = input[0][0][0].length;

        int outHeight = (height + 2 * pad - filterH) / stride + 1;
        int outWidth = (width + 2 * pad - filterW) / stride + 1;
        
        int outputRows = numSamples * outHeight * outWidth;
        int outputCols = channels * filterH * filterW;
        float[][] output = new float[outputRows][outputCols];
        
        // 并行处理每个样本
        java.util.stream.IntStream.range(0, numSamples).parallel().forEach(n -> {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    int outputRowIndex = (n * outHeight + h) * outWidth + w;
                    int colIndex = 0;
                    
                    for (int c = 0; c < channels; c++) {
                        for (int fh = 0; fh < filterH; fh++) {
                            int imRow = h * stride + fh - pad;
                            for (int fw = 0; fw < filterW; fw++) {
                                int imCol = w * stride + fw - pad;
                                
                                if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
                                    output[outputRowIndex][colIndex] = input[n][c][imRow][imCol];
                                } else {
                                    output[outputRowIndex][colIndex] = 0.0f;
                                }
                                colIndex++;
                            }
                        }
                    }
                }
            }
        });
        
        return output;
    }
    
    /**
     * 清理缓存
     */
    public static void clearCache() {
        outputCache.clear();
    }

}

