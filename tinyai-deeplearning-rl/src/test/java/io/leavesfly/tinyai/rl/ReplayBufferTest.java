package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * ReplayBuffer类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class ReplayBufferTest {

    private ReplayBuffer buffer;
    private final int CAPACITY = 5;

    @Before
    public void setUp() {
        buffer = new ReplayBuffer(CAPACITY);
    }

    /**
     * 创建测试Experience
     */
    private Experience createTestExperience(float stateValue, float actionValue, float reward, boolean done) {
        Variable state = new Variable(NdArray.of(new float[]{stateValue}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{actionValue}, Shape.of(1)));
        Variable nextState = new Variable(NdArray.of(new float[]{stateValue + 1.0f}, Shape.of(1)));
        return new Experience(state, action, reward, nextState, done);
    }

    /**
     * 测试初始状态
     */
    @Test
    public void testInitialState() {
        assertEquals(CAPACITY, buffer.getCapacity());
        assertEquals(0, buffer.size());
        assertTrue(buffer.isEmpty());
        assertFalse(buffer.isFull());
        assertEquals(0.0f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试单个经验添加
     */
    @Test
    public void testPushSingleExperience() {
        Experience experience = createTestExperience(1.0f, 0.0f, 1.0f, false);
        
        buffer.push(experience);
        
        assertEquals(1, buffer.size());
        assertFalse(buffer.isEmpty());
        assertFalse(buffer.isFull());
        assertEquals(0.2f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试多个经验添加
     */
    @Test
    public void testPushMultipleExperiences() {
        for (int i = 0; i < 3; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        assertEquals(3, buffer.size());
        assertFalse(buffer.isEmpty());
        assertFalse(buffer.isFull());
        assertEquals(0.6f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试缓冲区填满
     */
    @Test
    public void testBufferFull() {
        // 填满缓冲区
        for (int i = 0; i < CAPACITY; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        assertEquals(CAPACITY, buffer.size());
        assertFalse(buffer.isEmpty());
        assertTrue(buffer.isFull());
        assertEquals(1.0f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试缓冲区溢出（覆盖旧数据）
     */
    @Test
    public void testBufferOverflow() {
        // 填满缓冲区
        for (int i = 0; i < CAPACITY; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 添加额外的经验，应该覆盖最旧的
        Experience newExperience = createTestExperience(100.0f, 100.0f, 100.0f, true);
        buffer.push(newExperience);
        
        assertEquals(CAPACITY, buffer.size());
        assertTrue(buffer.isFull());
        assertEquals(1.0f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试采样功能
     */
    @Test
    public void testSampling() {
        // 添加一些经验
        for (int i = 0; i < 4; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 测试采样
        int batchSize = 2;
        Experience[] batch = buffer.sample(batchSize);
        
        assertEquals(batchSize, batch.length);
        assertNotNull(batch[0]);
        assertNotNull(batch[1]);
    }

    /**
     * 测试采样大小超过缓冲区大小
     */
    @Test(expected = IllegalArgumentException.class)
    public void testSamplingExceedsBufferSize() {
        // 只添加2个经验
        for (int i = 0; i < 2; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 尝试采样3个，应该抛出异常
        buffer.sample(3);
    }

    /**
     * 测试canSample方法
     */
    @Test
    public void testCanSample() {
        assertFalse(buffer.canSample(1)); // 空缓冲区
        
        // 添加2个经验
        for (int i = 0; i < 2; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        assertTrue(buffer.canSample(1));
        assertTrue(buffer.canSample(2));
        assertFalse(buffer.canSample(3));
    }

    /**
     * 测试随机采样的随机性
     */
    @Test
    public void testSamplingRandomness() {
        // 添加足够的经验
        for (int i = 0; i < CAPACITY; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 多次采样，验证结果不完全相同
        Set<Float> seenRewards = new HashSet<>();
        for (int trial = 0; trial < 10; trial++) {
            Experience[] batch = buffer.sample(1);
            seenRewards.add(batch[0].getReward());
        }
        
        // 应该看到多个不同的奖励值（由于随机性）
        assertTrue("采样应该具有随机性", seenRewards.size() > 1);
    }

    /**
     * 测试clear功能
     */
    @Test
    public void testClear() {
        // 添加一些经验
        for (int i = 0; i < 3; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        buffer.clear();
        
        assertEquals(0, buffer.size());
        assertTrue(buffer.isEmpty());
        assertFalse(buffer.isFull());
        assertEquals(0.0f, buffer.getUsageRate(), 0.001f);
    }

    /**
     * 测试getRecent功能
     */
    @Test
    public void testGetRecent() {
        // 添加经验
        for (int i = 0; i < 4; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 获取最近的2个经验
        List<Experience> recent = buffer.getRecent(2);
        
        assertEquals(2, recent.size());
        // 最近添加的应该是reward=3和reward=2
        assertEquals(3.0f, recent.get(0).getReward(), 0.001f);
        assertEquals(2.0f, recent.get(1).getReward(), 0.001f);
    }

    /**
     * 测试getRecent请求数量超过缓冲区大小
     */
    @Test
    public void testGetRecentExceedsSize() {
        // 只添加2个经验
        for (int i = 0; i < 2; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 请求5个，应该只返回2个
        List<Experience> recent = buffer.getRecent(5);
        assertEquals(2, recent.size());
    }

    /**
     * 测试循环覆盖后的getRecent
     */
    @Test
    public void testGetRecentAfterOverflow() {
        // 填满缓冲区
        for (int i = 0; i < CAPACITY; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 添加额外的经验触发覆盖
        for (int i = CAPACITY; i < CAPACITY + 2; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        // 获取最近的经验
        List<Experience> recent = buffer.getRecent(3);
        assertEquals(3, recent.size());
        
        // 最新的应该是reward=6, 5, 4
        assertEquals(6.0f, recent.get(0).getReward(), 0.001f);
        assertEquals(5.0f, recent.get(1).getReward(), 0.001f);
        assertEquals(4.0f, recent.get(2).getReward(), 0.001f);
    }

    /**
     * 测试toString方法
     */
    @Test
    public void testToString() {
        String emptyResult = buffer.toString();
        assertTrue(emptyResult.contains("0/5"));
        assertTrue(emptyResult.contains("0.00%"));
        
        // 添加一些经验
        for (int i = 0; i < 3; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            buffer.push(experience);
        }
        
        String result = buffer.toString();
        assertTrue(result.contains("3/5"));
        assertTrue(result.contains("60.00%"));
        assertTrue(result.contains("ReplayBuffer"));
    }

    /**
     * 测试边界情况：容量为1的缓冲区
     */
    @Test
    public void testMinimalCapacity() {
        ReplayBuffer minBuffer = new ReplayBuffer(1);
        
        Experience exp1 = createTestExperience(1.0f, 1.0f, 1.0f, false);
        Experience exp2 = createTestExperience(2.0f, 2.0f, 2.0f, false);
        
        minBuffer.push(exp1);
        assertEquals(1, minBuffer.size());
        assertTrue(minBuffer.isFull());
        
        minBuffer.push(exp2);
        assertEquals(1, minBuffer.size());
        assertTrue(minBuffer.isFull());
        
        // 采样应该返回最新的经验
        Experience[] batch = minBuffer.sample(1);
        assertEquals(2.0f, batch[0].getReward(), 0.001f);
    }

    /**
     * 测试大容量缓冲区
     */
    @Test
    public void testLargeCapacity() {
        ReplayBuffer largeBuffer = new ReplayBuffer(1000);
        
        assertEquals(1000, largeBuffer.getCapacity());
        assertTrue(largeBuffer.isEmpty());
        assertFalse(largeBuffer.isFull());
        
        // 添加一些经验
        for (int i = 0; i < 100; i++) {
            Experience experience = createTestExperience(i, i, i, false);
            largeBuffer.push(experience);
        }
        
        assertEquals(100, largeBuffer.size());
        assertEquals(0.1f, largeBuffer.getUsageRate(), 0.001f);
        assertFalse(largeBuffer.isFull());
    }
}