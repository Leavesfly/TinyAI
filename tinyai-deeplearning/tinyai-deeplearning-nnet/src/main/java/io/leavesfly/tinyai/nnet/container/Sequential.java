package io.leavesfly.tinyai.nnet.container;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.core.Module;

import java.util.ArrayList;
import java.util.List;

/**
 * V2版本的Sequential容器
 * <p>
 * Sequential按顺序组合多个模块，前一个模块的输出作为后一个模块的输入。
 * <p>
 * 特性：
 * - 支持链式调用添加模块
 * - 自动命名子模块（0, 1, 2...）
 * - 顺序前向传播
 *
 * 使用示例：
 * <pre>
 * Sequential model = new Sequential("model")
 *     .add(new Linear("fc1", 128, 64))
 *     .add(new ReLU())
 *     .add(new Linear("fc2", 64, 10));
 * </pre>
 *
 * @author leavesfly
 * @version 2.0
 */
public class Sequential extends Module {

    private final List<Module> moduleList;
    private int moduleCount = 0;

    /**
     * 构造函数
     *
     * @param name 容器名称
     */
    public Sequential(String name) {
        super(name);
        this.moduleList = new ArrayList<>();
    }

    /**
     * 默认构造函数
     */
    public Sequential() {
        this("sequential");
    }

    /**
     * 添加模块到容器
     * <p>
     * 支持链式调用
     *
     * @param module 要添加的模块
     * @return 当前Sequential对象（支持链式调用）
     */
    public Sequential add(Module module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot add null module to Sequential");
        }

        // 自动命名：使用索引作为名称
        String moduleName = String.valueOf(moduleCount);
        registerModule(moduleName, module);
        moduleList.add(module);
        moduleCount++;

        return this;
    }

    /**
     * 添加模块到容器（指定名称）
     *
     * @param name   模块名称
     * @param module 要添加的模块
     * @return 当前Sequential对象（支持链式调用）
     */
    public Sequential add(String name, Module module) {
        if (module == null) {
            throw new IllegalArgumentException("Cannot add null module to Sequential");
        }

        registerModule(name, module);
        moduleList.add(module);
        moduleCount++;

        return this;
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (moduleList.isEmpty()) {
            throw new IllegalStateException("Sequential is empty, no modules to forward");
        }

        if (inputs.length == 0) {
            throw new IllegalArgumentException("Sequential requires at least one input");
        }

        // 顺序前向传播
        Variable output = inputs[0];
        for (Module module : moduleList) {
            output = module.forward(output);
        }

        return output;
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
     * 获取模块数量
     *
     * @return 模块数量
     */
    public int size() {
        return moduleList.size();
    }

    /**
     * 判断容器是否为空
     *
     * @return true表示容器为空
     */
    public boolean isEmpty() {
        return moduleList.isEmpty();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Sequential{name='").append(name).append("', modules=[\n");
        for (int i = 0; i < moduleList.size(); i++) {
            sb.append("  (").append(i).append("): ").append(moduleList.get(i)).append("\n");
        }
        sb.append("]}");
        return sb.toString();
    }
}
