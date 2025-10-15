package io.leavesfly.tinyai.agent.cursor.v2.infra.storage;

import io.leavesfly.tinyai.agent.cursor.v2.model.Session;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 会话存储
 * 管理会话的持久化和检索
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class SessionStore {
    
    /**
     * 会话存储（sessionId -> Session）
     */
    private final Map<String, Session> sessions;
    
    public SessionStore() {
        this.sessions = new ConcurrentHashMap<>();
    }
    
    /**
     * 保存会话
     */
    public void save(Session session) {
        sessions.put(session.getSessionId(), session);
    }
    
    /**
     * 获取会话
     */
    public Session get(String sessionId) {
        return sessions.get(sessionId);
    }
    
    /**
     * 删除会话
     */
    public void delete(String sessionId) {
        sessions.remove(sessionId);
    }
    
    /**
     * 获取用户的所有会话
     */
    public List<Session> getUserSessions(String userId) {
        return sessions.values().stream()
                .filter(session -> userId.equals(session.getUserId()))
                .collect(Collectors.toList());
    }
    
    /**
     * 获取项目的所有会话
     */
    public List<Session> getProjectSessions(String projectId) {
        return sessions.values().stream()
                .filter(session -> projectId.equals(session.getProjectId()))
                .collect(Collectors.toList());
    }
    
    /**
     * 获取活跃会话
     */
    public List<Session> getActiveSessions() {
        return sessions.values().stream()
                .filter(Session::isActive)
                .collect(Collectors.toList());
    }
    
    /**
     * 清理过期会话
     */
    public int cleanupExpiredSessions() {
        List<String> expiredSessionIds = sessions.values().stream()
                .filter(Session::isExpired)
                .map(Session::getSessionId)
                .collect(Collectors.toList());
        
        expiredSessionIds.forEach(sessions::remove);
        return expiredSessionIds.size();
    }
    
    /**
     * 获取会话总数
     */
    public int size() {
        return sessions.size();
    }
    
    /**
     * 清空所有会话
     */
    public void clear() {
        sessions.clear();
    }
}
