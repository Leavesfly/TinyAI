package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArrayAccumulationTest - 测试累加操作
 * 
 * 测试通过 NdArrayCpu 门面调用的 AccumulationOperations 方法
 *
 * @author TinyAI
 */
public class NdArrayAccumulationTest {

    @Test
    public void testAddAtEqualLength() {
        // addAt 等长索引模式：3x4矩阵，在(0,1),(1,2)位置累加值
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        
        NdArrayCpu other = new NdArrayCpu(new float[]{10, 20});
        
        NdArrayCpu result = array.addAt(new int[]{0, 1}, new int[]{1, 2}, other);
        
        // 验证(0,1)位置累加10: 2 + 10 = 12
        assertEquals(12.0f, result.get(0, 1), 1e-6);
        // 验证(1,2)位置累加20: 7 + 20 = 27
        assertEquals(27.0f, result.get(1, 2), 1e-6);
        // 验证其他位置不变
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(5.0f, result.get(1, 0), 1e-6);
    }

    @Test
    public void testAddAtDifferentLength() {
        // addAt 不等长索引模式：3x4矩阵，rowSlices=[0,1], colSlices=[1,2,3]，对所有组合累加
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        
        NdArrayCpu other = new NdArrayCpu(new float[]{10, 20, 30, 40, 50, 60});
        
        NdArrayCpu result = array.addAt(new int[]{0, 1}, new int[]{1, 2, 3}, other);
        
        // 验证所有组合位置都被累加
        // (0,1): 2 + 10 = 12
        assertEquals(12.0f, result.get(0, 1), 1e-6);
        // (0,2): 3 + 20 = 23
        assertEquals(23.0f, result.get(0, 2), 1e-6);
        // (0,3): 4 + 30 = 34
        assertEquals(34.0f, result.get(0, 3), 1e-6);
        // (1,1): 6 + 40 = 46
        assertEquals(46.0f, result.get(1, 1), 1e-6);
        // (1,2): 7 + 50 = 57
        assertEquals(57.0f, result.get(1, 2), 1e-6);
        // (1,3): 8 + 60 = 68
        assertEquals(68.0f, result.get(1, 3), 1e-6);
        // 验证未被累加的位置不变
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(5.0f, result.get(1, 0), 1e-6);
        assertEquals(9.0f, result.get(2, 0), 1e-6);
    }

    @Test
    public void testAddAtNullRowSlices() {
        // addAt null行索引：展开为[0,1,2]，与colSlices=[0,1,2]等长
        // 走 addAtEqualLength 路径：(0,0)+=10, (1,1)+=20, (2,2)+=30
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        NdArrayCpu other = new NdArrayCpu(new float[]{10, 20, 30});

        NdArrayCpu result = array.addAt(null, new int[]{0, 1, 2}, other);

        // 对角线位置被累加
        assertEquals(11.0f, result.get(0, 0), 1e-6); // 1 + 10
        assertEquals(2.0f, result.get(0, 1), 1e-6);   // 不变
        assertEquals(3.0f, result.get(0, 2), 1e-6);   // 不变
        assertEquals(4.0f, result.get(1, 0), 1e-6);   // 不变
        assertEquals(25.0f, result.get(1, 1), 1e-6);  // 5 + 20
        assertEquals(6.0f, result.get(1, 2), 1e-6);   // 不变
        assertEquals(7.0f, result.get(2, 0), 1e-6);   // 不变
        assertEquals(8.0f, result.get(2, 1), 1e-6);   // 不变
        assertEquals(39.0f, result.get(2, 2), 1e-6);  // 9 + 30
    }

    @Test
    public void testAddAtNullColSlices() {
        // addAt null列索引：展开为[0,1,2]，与rowSlices=[0,1,2]等长
        // 走 addAtEqualLength 路径：(0,0)+=10, (1,1)+=20, (2,2)+=30
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        NdArrayCpu other = new NdArrayCpu(new float[]{10, 20, 30});

        NdArrayCpu result = array.addAt(new int[]{0, 1, 2}, null, other);

        // 对角线位置被累加
        assertEquals(11.0f, result.get(0, 0), 1e-6); // 1 + 10
        assertEquals(2.0f, result.get(0, 1), 1e-6);   // 不变
        assertEquals(3.0f, result.get(0, 2), 1e-6);   // 不变
        assertEquals(4.0f, result.get(1, 0), 1e-6);   // 不变
        assertEquals(25.0f, result.get(1, 1), 1e-6);  // 5 + 20
        assertEquals(6.0f, result.get(1, 2), 1e-6);   // 不变
        assertEquals(7.0f, result.get(2, 0), 1e-6);   // 不变
        assertEquals(8.0f, result.get(2, 1), 1e-6);   // 不变
        assertEquals(39.0f, result.get(2, 2), 1e-6);  // 9 + 30
    }

    @Test
    public void testAddToBasic() {
        // addTo 基本功能：3x4矩阵在(1,1)位置累加一个2x2子矩阵
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        });
        
        NdArrayCpu subMatrix = new NdArrayCpu(new float[][]{
            {10, 20},
            {30, 40}
        });
        
        NdArrayCpu result = array.addTo(1, 1, subMatrix);
        
        // 验证累加结果
        assertEquals(1.0f, result.get(0, 0), 1e-6);
        assertEquals(2.0f, result.get(0, 1), 1e-6);
        assertEquals(3.0f, result.get(0, 2), 1e-6);
        assertEquals(4.0f, result.get(0, 3), 1e-6);
        
        // 第一行被累加
        assertEquals(5.0f, result.get(1, 0), 1e-6);
        assertEquals(16.0f, result.get(1, 1), 1e-6); // 6 + 10
        assertEquals(27.0f, result.get(1, 2), 1e-6); // 7 + 20
        assertEquals(8.0f, result.get(1, 3), 1e-6);
        
        // 第二行被累加
        assertEquals(9.0f, result.get(2, 0), 1e-6);
        assertEquals(40.0f, result.get(2, 1), 1e-6); // 10 + 30
        assertEquals(51.0f, result.get(2, 2), 1e-6); // 11 + 40
        assertEquals(12.0f, result.get(2, 3), 1e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAddToOutOfBounds() {
        // addTo 边界检查：偏移导致超出范围时抛异常
        NdArrayCpu array = new NdArrayCpu(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        
        NdArrayCpu subMatrix = new NdArrayCpu(new float[][]{
            {10, 20},
            {30, 40}
        });
        
        // 在(1,2)位置累加2x2矩阵会超出范围
        array.addTo(1, 2, subMatrix);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAddToNonMatrix() {
        // addTo 非矩阵抛异常
        NdArrayCpu array = new NdArrayCpu(new float[]{1, 2, 3, 4});
        
        NdArrayCpu subMatrix = new NdArrayCpu(new float[][]{
            {10, 20},
            {30, 40}
        });
        
        array.addTo(0, 0, subMatrix);
    }
}
