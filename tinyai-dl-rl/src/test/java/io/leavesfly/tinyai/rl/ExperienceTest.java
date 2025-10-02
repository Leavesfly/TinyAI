package io.leavesfly.tinyai.rl;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Experience类的单元测试
 * 
 * @author leavesfly
 * @version 0.01
 */
public class ExperienceTest {

    /**
     * 测试Experience基本构造函数
     */
    @Test
    public void testBasicConstructor() {
        // 创建测试数据
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable action = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        float reward = 1.5f;
        Variable nextState = new Variable(NdArray.of(new float[]{2.0f, 3.0f}, Shape.of(1, 2)));
        boolean done = false;
        
        // 创建Experience实例
        Experience experience = new Experience(state, action, reward, nextState, done);
        
        // 验证属性
        assertEquals(state, experience.getState());
        assertEquals(action, experience.getAction());
        assertEquals(reward, experience.getReward(), 0.001f);
        assertEquals(nextState, experience.getNextState());
        assertEquals(done, experience.isDone());
        assertEquals(-1, experience.getTimeStep()); // 默认值
    }

    /**
     * 测试Experience完整构造函数
     */
    @Test
    public void testFullConstructor() {
        // 创建测试数据
        Variable state = new Variable(NdArray.of(new float[]{0.5f}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{1.0f}, Shape.of(1)));
        float reward = -0.5f;
        Variable nextState = new Variable(NdArray.of(new float[]{1.5f}, Shape.of(1)));
        boolean done = true;
        int timeStep = 42;
        
        // 创建Experience实例
        Experience experience = new Experience(state, action, reward, nextState, done, timeStep);
        
        // 验证属性
        assertEquals(state, experience.getState());
        assertEquals(action, experience.getAction());
        assertEquals(reward, experience.getReward(), 0.001f);
        assertEquals(nextState, experience.getNextState());
        assertEquals(done, experience.isDone());
        assertEquals(timeStep, experience.getTimeStep());
    }

    /**
     * 测试终止状态Experience
     */
    @Test
    public void testTerminalExperience() {
        Variable state = new Variable(NdArray.of(new float[]{5.0f}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{2.0f}, Shape.of(1)));
        float reward = 10.0f;
        Variable nextState = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1))); // 终止状态
        boolean done = true;
        
        Experience experience = new Experience(state, action, reward, nextState, done);
        
        assertTrue(experience.isDone());
        assertEquals(10.0f, experience.getReward(), 0.001f);
    }

    /**
     * 测试负奖励Experience
     */
    @Test
    public void testNegativeReward() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        float reward = -5.5f;
        Variable nextState = new Variable(NdArray.of(new float[]{0.5f}, Shape.of(1)));
        boolean done = false;
        
        Experience experience = new Experience(state, action, reward, nextState, done);
        
        assertEquals(-5.5f, experience.getReward(), 0.001f);
        assertFalse(experience.isDone());
    }

    /**
     * 测试多维状态和动作
     */
    @Test
    public void testMultiDimensionalData() {
        // 多维状态 (2x3)
        Variable state = new Variable(NdArray.of(new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }, Shape.of(2, 3)));
        
        // 多维动作 (1x4)
        Variable action = new Variable(NdArray.of(new float[]{
            0.1f, 0.2f, 0.3f, 0.4f
        }, Shape.of(1, 4)));
        
        float reward = 2.5f;
        
        // 多维下一状态
        Variable nextState = new Variable(NdArray.of(new float[]{
            1.1f, 2.1f, 3.1f,
            4.1f, 5.1f, 6.1f
        }, Shape.of(2, 3)));
        
        Experience experience = new Experience(state, action, reward, nextState, false, 100);
        
        // 验证形状保持不变
        assertEquals(Shape.of(2, 3), experience.getState().getValue().getShape());
        assertEquals(Shape.of(1, 4), experience.getAction().getValue().getShape());
        assertEquals(Shape.of(2, 3), experience.getNextState().getValue().getShape());
        assertEquals(100, experience.getTimeStep());
    }

    /**
     * 测试toString方法
     */
    @Test
    public void testToString() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{0.0f}, Shape.of(1)));
        float reward = 3.14159f;
        Variable nextState = new Variable(NdArray.of(new float[]{2.0f}, Shape.of(1)));
        boolean done = true;
        int timeStep = 25;
        
        Experience experience = new Experience(state, action, reward, nextState, done, timeStep);
        String result = experience.toString();
        
        // 验证toString包含关键信息
        assertTrue(result.contains("timeStep=25"));
        assertTrue(result.contains("reward=3.1416"));
        assertTrue(result.contains("done=true"));
        assertTrue(result.contains("Experience"));
    }

    /**
     * 测试边界值
     */
    @Test
    public void testBoundaryValues() {
        Variable state = new Variable(NdArray.of(new float[]{Float.MAX_VALUE}, Shape.of(1)));
        Variable action = new Variable(NdArray.of(new float[]{Float.MIN_VALUE}, Shape.of(1)));
        float reward = 0.0f;
        Variable nextState = new Variable(NdArray.of(new float[]{Float.NEGATIVE_INFINITY}, Shape.of(1)));
        
        Experience experience = new Experience(state, action, reward, nextState, false);
        
        assertEquals(Float.MAX_VALUE, experience.getState().getValue().get(0), 0.001f);
        assertEquals(Float.MIN_VALUE, experience.getAction().getValue().get(0), 0.001f);
        assertEquals(0.0f, experience.getReward(), 0.001f);
        assertEquals(Float.NEGATIVE_INFINITY, experience.getNextState().getValue().get(0), 0.001f);
    }

    /**
     * 测试相同实例的属性访问
     */
    @Test
    public void testConsistentAccess() {
        Variable state = new Variable(NdArray.of(new float[]{1.0f, 2.0f}, Shape.of(1, 2)));
        Variable action = new Variable(NdArray.of(new float[]{1.0f}, Shape.of(1)));
        float reward = 1.0f;
        Variable nextState = new Variable(NdArray.of(new float[]{2.0f, 3.0f}, Shape.of(1, 2)));
        
        Experience experience = new Experience(state, action, reward, nextState, false, 10);
        
        // 多次访问应该返回相同的值
        for (int i = 0; i < 5; i++) {
            assertEquals(state, experience.getState());
            assertEquals(action, experience.getAction());
            assertEquals(reward, experience.getReward(), 0.001f);
            assertEquals(nextState, experience.getNextState());
            assertFalse(experience.isDone());
            assertEquals(10, experience.getTimeStep());
        }
    }
}