package io.leavesfly.tinyai.agent.embodied.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 感知状态测试类
 *
 * @author TinyAI Team
 */
public class PerceptionStateTest {

    private PerceptionState state;

    @BeforeEach
    public void setUp() {
        state = new PerceptionState();
    }

    @Test
    public void testDefaultConstructor() {
        assertNotNull(state.getObstacleMap());
        assertTrue(state.getObstacleMap().isEmpty());
        assertTrue(state.getTimestamp() > 0);
    }

    @Test
    public void testSettersAndGetters() {
        VehicleState vehicleState = new VehicleState();
        vehicleState.setSpeed(25.0);
        state.setVehicleState(vehicleState);
        
        assertEquals(vehicleState, state.getVehicleState());
        assertEquals(25.0, state.getVehicleState().getSpeed(), 1e-6);
    }

    @Test
    public void testGetNearestObstacleEmpty() {
        ObstacleInfo nearest = state.getNearestObstacle();
        assertNull(nearest);
    }

    @Test
    public void testGetNearestObstacleSingle() {
        ObstacleInfo obstacle = createObstacle(50.0, 0.0);
        
        List<ObstacleInfo> obstacles = new ArrayList<>();
        obstacles.add(obstacle);
        state.setObstacleMap(obstacles);
        
        ObstacleInfo nearest = state.getNearestObstacle();
        assertNotNull(nearest);
        assertEquals(50.0, nearest.getDistance(), 1e-6);
    }

    @Test
    public void testGetNearestObstacleMultiple() {
        ObstacleInfo o1 = createObstacle(50.0, 0.0);
        ObstacleInfo o2 = createObstacle(30.0, 0.0);
        ObstacleInfo o3 = createObstacle(70.0, 0.0);
        
        List<ObstacleInfo> obstacles = new ArrayList<>();
        obstacles.add(o1);
        obstacles.add(o2);
        obstacles.add(o3);
        state.setObstacleMap(obstacles);
        
        ObstacleInfo nearest = state.getNearestObstacle();
        assertNotNull(nearest);
        assertEquals(30.0, nearest.getDistance(), 1e-6);
    }

    @Test
    public void testCountObstaclesInRange() {
        ObstacleInfo o1 = createObstacle(20.0, 0.0);
        ObstacleInfo o2 = createObstacle(40.0, 0.0);
        ObstacleInfo o3 = createObstacle(60.0, 0.0);
        
        List<ObstacleInfo> obstacles = new ArrayList<>();
        obstacles.add(o1);
        obstacles.add(o2);
        obstacles.add(o3);
        state.setObstacleMap(obstacles);
        
        assertEquals(2, state.countObstaclesInRange(50.0));
        assertEquals(1, state.countObstaclesInRange(30.0));
        assertEquals(3, state.countObstaclesInRange(70.0));
        assertEquals(0, state.countObstaclesInRange(10.0));
    }

    @Test
    public void testGetDangerousObstacles() {
        ObstacleInfo o1 = createObstacle(15.0, 5.0);
        ObstacleInfo o2 = createObstacle(50.0, 3.0);
        ObstacleInfo o3 = createObstacle(18.0, 8.0);
        
        List<ObstacleInfo> obstacles = new ArrayList<>();
        obstacles.add(o1);
        obstacles.add(o2);
        obstacles.add(o3);
        state.setObstacleMap(obstacles);
        
        List<ObstacleInfo> dangerous = state.getDangerousObstacles(20.0, 4.0);
        
        assertEquals(2, dangerous.size());
        assertTrue(dangerous.contains(o1));
        assertTrue(dangerous.contains(o3));
    }

    @Test
    public void testTimestamp() {
        long timestamp = System.currentTimeMillis();
        state.setTimestamp(timestamp);
        
        assertEquals(timestamp, state.getTimestamp());
    }

    @Test
    public void testToString() {
        VehicleState vehicleState = new VehicleState();
        vehicleState.setSpeed(30.0);
        state.setVehicleState(vehicleState);
        
        ObstacleInfo obstacle = createObstacle(10.0, 0.0);
        List<ObstacleInfo> obstacles = new ArrayList<>();
        obstacles.add(obstacle);
        state.setObstacleMap(obstacles);
        
        String str = state.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("PerceptionState"));
        assertTrue(str.contains("obstacles=1"));
    }

    @Test
    public void testSetObstacleMap() {
        List<ObstacleInfo> obstacles = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ObstacleInfo obstacle = createObstacle(i * 10.0, 0.0);
            obstacles.add(obstacle);
        }
        
        state.setObstacleMap(obstacles);
        
        assertEquals(5, state.getObstacleMap().size());
    }

    /**
     * 辅助方法：创建障碍物对象
     */
    private ObstacleInfo createObstacle(double distance, double velocity) {
        Vector3D position = new Vector3D(distance, 0.0, 0.0);
        Vector3D vel = new Vector3D(velocity, 0.0, 0.0);
        BoundingBox bbox = new BoundingBox(4.0, 2.0, 1.5);
        return new ObstacleInfo(ObstacleType.VEHICLE, position, vel, bbox, 0.9);
    }
}
