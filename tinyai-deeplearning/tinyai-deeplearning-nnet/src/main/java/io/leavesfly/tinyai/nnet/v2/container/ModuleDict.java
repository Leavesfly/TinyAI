package io.leavesfly.tinyai.nnet.v2.container;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * 字典形式存放子模块的容器，行为与 PyTorch 的 ModuleDict 类似。
 * <p>
 * 支持按键名增删改查，保留插入顺序；不定义前向逻辑，需在自定义 forward 中手动调用。
 */
public class ModuleDict extends Module implements Iterable<Module> {

    public ModuleDict(String name) {
        super(name);
    }

    public ModuleDict() {
        this("module_dict");
    }

    /**
     * 新增或替换键对应的子模块。
     *
     * @param key    子模块名称
     * @param module 子模块实例
     * @return 被替换掉的旧模块；若为新增则返回 null
     */
    public Module put(String key, Module module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot put null module into ModuleDict");
        }

        Module old = _modules.get(key);
        if (old == null) {
            registerModule(key, module);
        } else {
            old.setParent(null);  // 清理旧模块的 parent 引用
            _modules.put(key, module);
            module.setParent(this);
        }
        return old;
    }

    /**
     * 按键获取子模块。
     */
    public Module get(String key) {
        return _modules.get(key);
    }

    /**
     * 删除并返回对应键的子模块。
     */
    public Module remove(String key) {
        Module removed = _modules.remove(key);
        if (removed != null) {
            removed.setParent(null);  // 清理被移除模块的 parent 引用
        }
        return removed;
    }

    public boolean containsKey(String key) {
        return _modules.containsKey(key);
    }

    public int size() {
        return _modules.size();
    }

    public boolean isEmpty() {
        return _modules.isEmpty();
    }

    public Set<String> keys() {
        return _modules.keySet();
    }

    public Collection<Module> values() {
        return _modules.values();
    }

    public Set<Map.Entry<String, Module>> entries() {
        return _modules.entrySet();
    }

    @Override
    public Iterator<Module> iterator() {
        return _modules.values().iterator();
    }

    @Override
    public Variable forward(Variable... inputs) {
        throw new UnsupportedOperationException(
                "ModuleDict does not define forward(). " +
                        "Use contained modules explicitly in your custom forward() method.");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ModuleDict{name='").append(name).append("', modules={\n");
        for (Map.Entry<String, Module> entry : _modules.entrySet()) {
            sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        sb.append("}}");
        return sb.toString();
    }
}

