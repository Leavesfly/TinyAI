package io.leavesfly.tinyai.nnet.v2.container;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * V2版本的ModuleList容器
 * <p>
 * ModuleList存储模块列表，支持索引访问和迭代。
 * 与Sequential不同，ModuleList不定义前向传播逻辑，需要用户自行组合。
 * <p>
 * 使用场景：
 * - 需要灵活组合模块的场景
 * - 重复的层结构（如Transformer的多个Encoder层）
 * - 需要在forward中动态选择模块
 * <p>
 * 使用示例：
 * <pre>
 * ModuleList layers = new ModuleList("layers");
 * for (int i = 0; i < 6; i++) {
 *     layers.add(new TransformerEncoderBlock("layer" + i, ...));
 * }
 *
 * // 在forward中使用
 * Variable x = input;
 * for (int i = 0; i < layers.size(); i++) {
 *     x = layers.get(i).forward(x);
 * }
 * </pre>
 *
 * @author leavesfly
 * @version 2.0
 */
public class ModuleList extends Module implements Iterable<Module> {

    private final List<Module> moduleList;

    /**
     * 构造函数
     *
     * @param name 容器名称
     */
    public ModuleList(String name) {
        super(name);
        this.moduleList = new ArrayList<>();
    }

    /**
     * 默认构造函数
     */
    public ModuleList() {
        this("module_list");
    }

    /**
     * 添加模块到列表
     *
     * @param module 要添加的模块
     * @return 当前ModuleList对象（支持链式调用）
     */
    public ModuleList add(Module module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot add null module to ModuleList");
        }

        int index = moduleList.size();
        String moduleName = String.valueOf(index);
        registerModule(moduleName, module);
        moduleList.add(module);

        return this;
    }

    /**
     * 获取指定索引的模块
     *
     * @param index 模块索引
     * @return 模块对象
     * @throws IndexOutOfBoundsException 索引超出范围
     */
    public Module get(int index) {
        return moduleList.get(index);
    }

    /**
     * 设置指定索引的模块
     *
     * @param index  模块索引
     * @param module 新模块
     * @return 被替换的旧模块
     */
    public Module set(int index, Module module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot set null module in ModuleList");
        }

        Module oldModule = moduleList.set(index, module);

        // 清理旧模块的 parent 引用
        if (oldModule != null) {
            oldModule.setParent(null);
        }

        // 更新注册
        String moduleName = String.valueOf(index);
        _modules.put(moduleName, module);
        module.setParent(this);

        return oldModule;
    }

    /**
     * 获取模块数量
     *
     * @return 模块数量
     */
    public int size() {
        return moduleList.size();
    }

    /**
     * 判断列表是否为空
     *
     * @return true表示列表为空
     */
    public boolean isEmpty() {
        return moduleList.isEmpty();
    }

    /**
     * 返回迭代器
     *
     * @return 模块迭代器
     */
    @Override
    public Iterator<Module> iterator() {
        return moduleList.iterator();
    }

    /**
     * ModuleList不定义默认的前向传播逻辑
     * <p>
     * 用户需要在自定义的forward方法中使用ModuleList
     *
     * @param inputs 输入变量
     * @return 抛出UnsupportedOperationException
     */
    @Override
    public Variable forward(Variable... inputs) {
        throw new UnsupportedOperationException(
                "ModuleList does not define forward(). " +
                        "Use modules individually in your custom forward() method.");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ModuleList{name='").append(name).append("', size=").append(size()).append(", modules=[\n");
        for (int i = 0; i < moduleList.size(); i++) {
            sb.append("  (").append(i).append("): ").append(moduleList.get(i)).append("\n");
        }
        sb.append("]}");
        return sb.toString();
    }
}
