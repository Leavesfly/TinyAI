package io.leavesfly.tinyai.agent.cursor.v2.service;

import io.leavesfly.tinyai.agent.cursor.v2.model.Session;
import io.leavesfly.tinyai.agent.cursor.v2.infra.storage.SessionStore;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 会话服务
 * 管理用户会话的生命周期
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class SessionService {
    
    private final SessionStore sessionStore;
    private final Map<String, Session> activeSessions;
    
    public SessionService(SessionStore sessionStore) {
        this.sessionStore = sessionStore;
        this.activeSessions = new ConcurrentHashMap<>();
    }
    
    public SessionService() {
        this(new SessionStore());
    }
    
    /**
     * 创建会话
     */
    public String createSession(String userId, String projectId) {
        Session session = new Session();
        session.setUserId(userId);
        session.setProjectId(projectId);
        
        activeSessions.put(session.getSessionId(), session);
        sessionStore.save(session);
        
        return session.getSessionId();
    }
    
    /**
     * 获取会话信息
     */
    public SessionInfo getSessionInfo(String sessionId) {
        Session session = activeSessions.get(sessionId);
        if (session == null) {
            session = sessionStore.get(sessionId);
        }
        
        if (session == null) {
            return null;
        }
        
        return convertToSessionInfo(session);
    }
    
    /**
     * 查找活跃会话
     */
    public String findActiveSession(String userId, String projectId) {
        for (Session session : activeSessions.values()) {
            if (userId.equals(session.getUserId()) && 
                projectId.equals(session.getProjectId()) &&
                !session.isExpired()) {
                return session.getSessionId();
            }
        }
        return null;
    }
    
    /**
     * 删除会话
     */
    public void deleteSession(String sessionId) {
        activeSessions.remove(sessionId);
        sessionStore.delete(sessionId);
    }
    
    /**
     * 添加分析历史
     */
    public void addAnalysisHistory(String sessionId, Object request, Object result) {
        Session session = activeSessions.get(sessionId);
        if (session != null) {
            session.updateActiveTime();
            // TODO: 保存历史记录
        }
    }
    
    /**
     * 获取统计信息
     */
    public SessionStats getStats() {
        SessionStats stats = new SessionStats();
        stats.totalSessions = activeSessions.size();
        stats.activeSessions = (int) activeSessions.values().stream()
            .filter(Session::isActive).count();
        return stats;
    }
    
    private SessionInfo convertToSessionInfo(Session session) {
        SessionInfo info = new SessionInfo();
        info.setSessionId(session.getSessionId());
        info.setUserId(session.getUserId());
        info.setProjectId(session.getProjectId());
        info.setCreatedAt(session.getCreatedAt());
        info.setLastAccessAt(session.getLastActiveAt());
        info.setMessageCount(session.getMessages().size());
        return info;
    }
    
    /**
     * 会话信息
     */
    public static class SessionInfo {
        private String sessionId;
        private String userId;
        private String projectId;
        private long createdAt;
        private long lastAccessAt;
        private int messageCount;
        
        public String getSessionId() { return sessionId; }
        public void setSessionId(String sessionId) { this.sessionId = sessionId; }
        public String getUserId() { return userId; }
        public void setUserId(String userId) { this.userId = userId; }
        public String getProjectId() { return projectId; }
        public void setProjectId(String projectId) { this.projectId = projectId; }
        public long getCreatedAt() { return createdAt; }
        public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }
        public long getLastAccessAt() { return lastAccessAt; }
        public void setLastAccessAt(long lastAccessAt) { this.lastAccessAt = lastAccessAt; }
        public int getMessageCount() { return messageCount; }
        public void setMessageCount(int messageCount) { this.messageCount = messageCount; }
    }
    
    /**
     * 会话统计
     */
    public static class SessionStats {
        public int totalSessions;
        public int activeSessions;
        
        @Override
        public String toString() {
            return "SessionStats{total=" + totalSessions + ", active=" + activeSessions + '}';
        }
    }
}
