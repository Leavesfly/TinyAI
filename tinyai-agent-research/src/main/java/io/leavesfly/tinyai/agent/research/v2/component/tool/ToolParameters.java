package io.leavesfly.tinyai.agent.research.v2.component.tool;

import java.util.*;

/**
 * 工具参数定义
 * 
 * @author TinyAI
 * @since 2.0.0
 */
public class ToolParameters {
    
    private final List<Parameter> parameters;
    
    public ToolParameters() {
        this.parameters = new ArrayList<>();
    }
    
    public void addParameter(Parameter parameter) {
        this.parameters.add(parameter);
    }
    
    public List<Parameter> getParameters() {
        return parameters;
    }
    
    /**
     * 参数定义
     */
    public static class Parameter {
        private final String name;
        private final String type;
        private final String description;
        private final boolean required;
        private final Object defaultValue;
        
        public Parameter(String name, String type, String description, boolean required) {
            this(name, type, description, required, null);
        }
        
        public Parameter(String name, String type, String description, boolean required, Object defaultValue) {
            this.name = name;
            this.type = type;
            this.description = description;
            this.required = required;
            this.defaultValue = defaultValue;
        }
        
        public String getName() { return name; }
        public String getType() { return type; }
        public String getDescription() { return description; }
        public boolean isRequired() { return required; }
        public Object getDefaultValue() { return defaultValue; }
    }
}
