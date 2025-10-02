package io.leavesfly.tinyai.ndarr.tpu;

import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Shape Tpu 版本
 * //todo
 */
public class ShapeTpu implements Shape {
    @Override
    public int getRow() {
        return 0;
    }

    @Override
    public int getColumn() {
        return 0;
    }

    @Override
    public boolean isMatrix() {
        return false;
    }

    @Override
    public boolean isScalar() {
        return false;
    }

    @Override
    public boolean isVector() {
        return false;
    }

    @Override
    public int size() {
        return 0;
    }

    @Override
    public int getIndex(int... indices) {
        return 0;
    }

    @Override
    public int getDimension(int dimIndex) {
        return 0;
    }

    @Override
    public int getDimNum() {
        return 0;
    }
}
