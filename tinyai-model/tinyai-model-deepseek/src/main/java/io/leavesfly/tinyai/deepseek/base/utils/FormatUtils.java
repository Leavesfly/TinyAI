package io.leavesfly.tinyai.deepseek.base.utils;

/**
 * 通用格式化工具类
 *
 * <p>统一 DeepSeek 系列（及可复用给其他模型）的常用格式化逻辑，
 * 避免在 Config/Demo/Monitor 等处重复实现同一套 formatParamCount / formatTime / formatRatio。
 *
 * <p>所有方法均为无状态纯函数，线程安全。
 *
 * @author leavesfly
 * @version 1.0
 */
public final class FormatUtils {

    private static final long ONE_K = 1_000L;
    private static final long ONE_M = 1_000_000L;
    private static final long ONE_B = 1_000_000_000L;

    private FormatUtils() {
        // 工具类禁止实例化
    }

    /**
     * 格式化参数数量为可读字符串（带单位后缀）
     *
     * <p>规则：
     * <ul>
     *   <li>{@code count >= 1e9} → "X.XXB"</li>
     *   <li>{@code count >= 1e6} → "X.XXM"</li>
     *   <li>{@code count >= 1e3} → "X.XXK"</li>
     *   <li>否则 → 原始数字（带千位分隔符）</li>
     * </ul>
     *
     * @param count 参数数量（允许 0，负数按绝对值处理）
     * @return 格式化字符串
     */
    public static String formatParamCount(long count) {
        long abs = Math.abs(count);
        String sign = count < 0 ? "-" : "";
        if (abs >= ONE_B) {
            return sign + String.format("%.2fB", abs / (double) ONE_B);
        }
        if (abs >= ONE_M) {
            return sign + String.format("%.2fM", abs / (double) ONE_M);
        }
        if (abs >= ONE_K) {
            return sign + String.format("%.2fK", abs / (double) ONE_K);
        }
        return sign + String.format("%,d", abs);
    }

    /**
     * 格式化毫秒时间为可读字符串
     *
     * <p>规则：
     * <ul>
     *   <li>{@code ms >= 3600_000} → "Xh Ym Zs"</li>
     *   <li>{@code ms >= 60_000} → "Xm Ys"</li>
     *   <li>否则 → "X.XXs"</li>
     * </ul>
     *
     * @param milliseconds 毫秒时长（允许 0）
     * @return 格式化字符串
     */
    public static String formatTime(long milliseconds) {
        long seconds = milliseconds / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;

        if (hours > 0) {
            return String.format("%dh %dm %ds", hours, minutes % 60, seconds % 60);
        }
        if (minutes > 0) {
            return String.format("%dm %ds", minutes, seconds % 60);
        }
        return String.format("%.2fs", milliseconds / 1000.0);
    }

    /**
     * 格式化比率为百分比字符串
     *
     * @param ratio 比率值（0.25 → "25.00%"）
     * @return 百分比字符串
     */
    public static String formatRatio(double ratio) {
        return String.format("%.2f%%", ratio * 100);
    }
}
