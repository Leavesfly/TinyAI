package io.leavesfly.tinyai.deepseek.base.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * FormatUtils 单元测试
 *
 * <p>验证 formatParamCount/formatTime/formatRatio 在边界值、负数、零、
 * 以及 K/M/B 不同量级下的格式化正确性。
 *
 * @author leavesfly
 */
public class FormatUtilsTest {

    @Test
    public void testFormatParamCount_belowOneThousand() {
        assertEquals("0", FormatUtils.formatParamCount(0L));
        assertEquals("1", FormatUtils.formatParamCount(1L));
        assertEquals("999", FormatUtils.formatParamCount(999L));
    }

    @Test
    public void testFormatParamCount_thousands() {
        assertEquals("1.00K", FormatUtils.formatParamCount(1_000L));
        assertEquals("1.23K", FormatUtils.formatParamCount(1_234L));
        assertEquals("999.99K", FormatUtils.formatParamCount(999_994L));
    }

    @Test
    public void testFormatParamCount_millions() {
        assertEquals("1.00M", FormatUtils.formatParamCount(1_000_000L));
        assertEquals("4.82M", FormatUtils.formatParamCount(4_819_544L));
        assertEquals("211.12M", FormatUtils.formatParamCount(211_121_712L));
    }

    @Test
    public void testFormatParamCount_billions() {
        assertEquals("1.00B", FormatUtils.formatParamCount(1_000_000_000L));
        assertEquals("1.23B", FormatUtils.formatParamCount(1_230_000_000L));
        assertEquals("671.00B", FormatUtils.formatParamCount(671_000_000_000L));
    }

    @Test
    public void testFormatParamCount_negative() {
        // 负数按绝对值格式化后带负号前缀
        assertEquals("-1.00K", FormatUtils.formatParamCount(-1_000L));
        assertEquals("-4.82M", FormatUtils.formatParamCount(-4_819_544L));
        assertEquals("-42", FormatUtils.formatParamCount(-42L));
    }

    @Test
    public void testFormatTime_subSecond() {
        assertEquals("0.00s", FormatUtils.formatTime(0L));
        assertEquals("0.50s", FormatUtils.formatTime(500L));
        assertEquals("1.00s", FormatUtils.formatTime(1_000L));
    }

    @Test
    public void testFormatTime_minutes() {
        assertEquals("1m 0s", FormatUtils.formatTime(60_000L));
        assertEquals("1m 30s", FormatUtils.formatTime(90_000L));
        assertEquals("59m 59s", FormatUtils.formatTime(3_599_000L));
    }

    @Test
    public void testFormatTime_hours() {
        assertEquals("1h 0m 0s", FormatUtils.formatTime(3_600_000L));
        assertEquals("2h 30m 15s", FormatUtils.formatTime(2L * 3_600_000L + 30L * 60_000L + 15_000L));
    }

    @Test
    public void testFormatRatio() {
        assertEquals("0.00%", FormatUtils.formatRatio(0.0));
        assertEquals("25.00%", FormatUtils.formatRatio(0.25));
        assertEquals("100.00%", FormatUtils.formatRatio(1.0));
        assertEquals("77.30%", FormatUtils.formatRatio(0.773));
    }
}
