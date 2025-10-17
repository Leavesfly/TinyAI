package io.leavesfly.tinyai.agent.embodied.decision;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.ObstacleInfo;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;

/**
 * 安全约束检查器
 */
public class SafetyConstraint {
    private static final double MIN_SAFE_DISTANCE = 10.0;
    private static final double MAX_STEERING = 0.5;
    
    public DrivingAction check(DrivingAction action, PerceptionState state) {
        DrivingAction safeAction = new DrivingAction(
            action.getSteering(), 
            action.getThrottle(), 
            action.getBrake()
        );
        
        // 1. 碰撞预防
        ObstacleInfo nearest = state.getNearestObstacle();
        if (nearest != null && nearest.getDistance() < MIN_SAFE_DISTANCE) {
            safeAction.setThrottle(0);
            safeAction.setBrake(0.8);
        }
        
        // 2. 转向限制
        if (Math.abs(safeAction.getSteering()) > MAX_STEERING) {
            safeAction.setSteering(Math.signum(safeAction.getSteering()) * MAX_STEERING);
        }
        
        // 3. 限幅
        safeAction.clip();
        
        return safeAction;
    }
}
