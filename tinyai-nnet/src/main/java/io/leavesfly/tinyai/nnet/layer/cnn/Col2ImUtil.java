package io.leavesfly.tinyai.nnet.layer.cnn;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 优化的Col2Im工具类
 * 增加了缓存机制和性能优化
 */
public class Col2ImUtil {
    
    // 缓存机制
    private static final Map<String, float[][][][]> outputCache = new ConcurrentHashMap<>();
    private static final int MAX_CACHE_SIZE = 5;
    
    /**
     * 生成缓存键
     */
    private static String generateCacheKey(int[] imgShape) {
        return String.format("%d_%d_%d_%d", imgShape[0], imgShape[1], imgShape[2], imgShape[3]);
    }
    
    /**
     * 获取或创建输出数组
     */
    private static float[][][][] getOrCreateOutput(String cacheKey, int[] imgShape, int pad) {
        float[][][][] output = outputCache.get(cacheKey);
        int N = imgShape[0], C = imgShape[1], H = imgShape[2], W = imgShape[3];
        
        if (output == null || output.length != N || output[0].length != C || 
            output[0][0].length != (H + 2 * pad) || output[0][0][0].length != (W + 2 * pad)) {
            output = new float[N][C][H + 2 * pad][W + 2 * pad];
            if (outputCache.size() >= MAX_CACHE_SIZE) {
                outputCache.clear();
            }
            outputCache.put(cacheKey, output);
        } else {
            // 清零复用的数组
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H + 2 * pad; h++) {
                        java.util.Arrays.fill(output[n][c][h], 0.0f);
                    }
                }
            }
        }
        return output;
    }

    /**
     * 优化版本：将列格式的数据还原成多维图像数组。
     * 增加了缓存机制和性能优化
     *
     * @param col      列格式的数据
     * @param imgShape 原始图像数据的形状，形式为 [N, C, H, W]。
     * @param filterH  滤波器的高度。
     * @param filterW  滤波器的宽度。
     * @param stride   步长。
     * @param pad      填充的大小。
     * @return 还原后的图像数据，形状为 [N, C, H, W]。
     */
    public static float[][][][] col2im(float[][] col, int[] imgShape, int filterH, int filterW, int stride, int pad) {
        int N = imgShape[0], C = imgShape[1], H = imgShape[2], W = imgShape[3];
        int outH = (H + 2 * pad - filterH) / stride + 1;
        int outW = (W + 2 * pad - filterW) / stride + 1;

        // 使用缓存机制获取输出数组
        String cacheKey = generateCacheKey(imgShape);
        float[][][][] img = getOrCreateOutput(cacheKey, imgShape, pad);

        // 优化的循环顺序
        for (int i = 0; i < N; i++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        // 定位到当前列在col中的位置
                        int colIndex = (c * outH + h) * outW + w;
                        int nOffset = i * C * outH * outW;
                        float[] kernel = col[nOffset + colIndex];

                        // 遍历滤波器的每个元素
                        for (int fh = 0; fh < filterH; fh++) {
                            int imgH = h * stride + fh;
                            for (int fw = 0; fw < filterW; fw++) {
                                int imgW = w * stride + fw;
                                
                                // 直接累加，避免边界检查（因为我们已经用填充创建了数组）
                                img[i][c][imgH][imgW] += kernel[fh * filterW + fw];
                            }
                        }
                    }
                }
            }
        }

        // 如果有填充，去除填充
        if (pad > 0) {
            return cropPadding(img, pad, H, W);
        } else {
            return img;
        }
    }
    
    /**
     * 高效版本：使用并行处理的更快版本
     */
    public static float[][][][] col2imFast(float[][] col, int[] imgShape, int filterH, int filterW, int stride, int pad) {
        int N = imgShape[0], C = imgShape[1], H = imgShape[2], W = imgShape[3];
        int outH = (H + 2 * pad - filterH) / stride + 1;
        int outW = (W + 2 * pad - filterW) / stride + 1;

        float[][][][] img = new float[N][C][H + 2 * pad][W + 2 * pad];

        // 并行处理每个样本
        java.util.stream.IntStream.range(0, N).parallel().forEach(i -> {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < outH; h++) {
                    for (int w = 0; w < outW; w++) {
                        int colIndex = (c * outH + h) * outW + w;
                        int nOffset = i * C * outH * outW;
                        float[] kernel = col[nOffset + colIndex];

                        for (int fh = 0; fh < filterH; fh++) {
                            int imgH = h * stride + fh;
                            for (int fw = 0; fw < filterW; fw++) {
                                int imgW = w * stride + fw;
                                img[i][c][imgH][imgW] += kernel[fh * filterW + fw];
                            }
                        }
                    }
                }
            }
        });

        if (pad > 0) {
            return cropPadding(img, pad, H, W);
        } else {
            return img;
        }
    }

    /**
     * 优化版本：从有填充的图像数组中裁剪出无填充的图像数组。
     *
     * @param img 包含填充的图像数组。
     * @param pad 填充的大小。
     * @param H   原始图像高度。
     * @param W   原始图像宽度。
     * @return 裁剪后的无填充的图像数组。
     */
    private static float[][][][] cropPadding(float[][][][] img, int pad, int H, int W) {
        int N = img.length;   // 样本数量（batch size）
        int C = img[0].length; // 通道数量
        
        // 创建一个不包含填充的数组，用来存储最终的结果
        float[][][][] result = new float[N][C][H][W];

        // 并行处理每个样本
        java.util.stream.IntStream.range(0, N).parallel().forEach(n -> {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    // 使用System.arraycopy进行快速复制
                    System.arraycopy(img[n][c][h + pad], pad, result[n][c][h], 0, W);
                }
            }
        });
        
        return result;
    }
    
    /**
     * 清理缓存
     */
    public static void clearCache() {
        outputCache.clear();
    }
}

