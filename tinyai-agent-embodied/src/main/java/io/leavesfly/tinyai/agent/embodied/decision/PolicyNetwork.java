package io.leavesfly.tinyai.agent.embodied.decision;

import io.leavesfly.tinyai.agent.embodied.model.DrivingAction;
import io.leavesfly.tinyai.agent.embodied.model.PerceptionState;

/**
 * 策略网络接口
 */
public interface PolicyNetwork {
    DrivingAction predict(PerceptionState state);
}

/**
 * 简单策略实现 - 保持车道和速度
 */
class SimplePolicy implements PolicyNetwork {
    @Override
    public DrivingAction predict(PerceptionState state) {
        DrivingAction action = new DrivingAction();
        
        if (state.getLaneInfo() != null) {
            // 根据车道偏离调整转向
            double deviation = state.getLaneInfo().getLateralDeviation();
            action.setSteering(-deviation * 0.1);
        }
        
        if (state.getVehicleState() != null) {
            double currentSpeed = state.getVehicleState().getSpeed();
            double targetSpeed = 22.0; // 约80 km/h
            
            if (currentSpeed < targetSpeed) {
                action.setThrottle(0.3);
            } else {
                action.setBrake(0.1);
            }
        }
        
        return action;
    }
}
