package io.leavesfly.tinyai.ndarr.tpu;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;

/**
 * NdArray的 Tpu 版本
 * //todo
 */

public class NdArrayTpu implements NdArray {

    @Override
    public NdArray like(Number value) {
        return null;
    }

    @Override
    public NdArrayCpu add(NdArray other) {
        return null;
    }

    @Override
    public NdArray sub(NdArray other) {
        return null;
    }

    @Override
    public NdArray mul(NdArray other) {
        return null;
    }

    @Override
    public NdArray mulNum(Number number) {
        return null;
    }

    @Override
    public NdArray div(NdArray other) {
        return null;
    }

    @Override
    public NdArray divNum(Number number) {
        return null;
    }

    @Override
    public NdArray neg() {
        return null;
    }

    @Override
    public NdArray abs() {
        return null;
    }

    @Override
    public NdArray eq(NdArray other) {
        return null;
    }

    @Override
    public NdArray gt(NdArray other) {
        return null;
    }

    @Override
    public NdArray lt(NdArray other) {
        return null;
    }

    @Override
    public boolean isLar(NdArray other) {
        return false;
    }

    @Override
    public NdArray pow(Number number) {
        return null;
    }

    @Override
    public NdArray square() {
        return null;
    }

    @Override
    public NdArray sqrt() {
        return null;
    }

    @Override
    public NdArray exp() {
        return null;
    }

    @Override
    public NdArray sin() {
        return null;
    }

    @Override
    public NdArray cos() {
        return null;
    }

    @Override
    public NdArray tanh() {
        return null;
    }

    @Override
    public NdArray sigmoid() {
        return null;
    }

    @Override
    public NdArray log() {
        return null;
    }

    @Override
    public NdArray softMax() {
        return null;
    }

    @Override
    public NdArray maximum(Number number) {
        return null;
    }

    @Override
    public NdArray mask(Number number) {
        return null;
    }

    @Override
    public NdArray transpose() {
        return null;
    }

    @Override
    public NdArray transpose(int... order) {
        return null;
    }

    @Override
    public NdArray reshape(Shape newShape) {
        return null;
    }

    @Override
    public NdArray flatten() {
        return null;
    }

    @Override
    public NdArray sum() {
        return null;
    }

    @Override
    public NdArray mean(int axis) {
        return null;
    }

    @Override
    public NdArray var(int axis) {
        return null;
    }

    @Override
    public NdArray sum(int axis) {
        return null;
    }

    @Override
    public NdArray sumTo(Shape _shape) {
        return null;
    }

    @Override
    public NdArray broadcastTo(Shape _shape) {
        return null;
    }

    @Override
    public NdArray argMax(int axis) {
        return null;
    }

    @Override
    public NdArray dot(NdArray other) {
        return null;
    }

    @Override
    public NdArray getItem(int[] _rowSlices, int[] _colSlices) {
        return null;
    }

    @Override
    public NdArray setItem(int[] _rowSlices, int[] _colSlices, float[] data) {
        return null;
    }

    @Override
    public NdArray max(int axis) {
        return null;
    }

    @Override
    public NdArray min(int axis) {
        return null;
    }

    @Override
    public float max() {
        return 0;
    }

    @Override
    public NdArray subNdArray(int startRow, int endRow, int startCol, int endCol) {
        return null;
    }

    @Override
    public NdArray addAt(int[] rowSlices, int[] colSlices, NdArray other) {
        return null;
    }

    @Override
    public NdArray addTo(int i, int j, NdArray other) {
        return null;
    }

    @Override
    public NdArray clip(float min, float max) {
        return null;
    }

    @Override
    public Number getNumber() {
        return null;
    }

    @Override
    public Shape getShape() {
        return null;
    }

    @Override
    public void setShape(Shape shape) {

    }

    @Override
    public float[] getArray() {
        return new float[0];
    }

    @Override
    public float[][] getMatrix() {
        return new float[0][];
    }

    @Override
    public float[][][] get3dArray() {
        return new float[0][][];
    }

    @Override
    public float[][][][] get4dArray() {
        return new float[0][][][];
    }

    @Override
    public void set(float value, int... _dimension) {

    }

    @Override
    public float get(int... _dimension) {
        return 0;
    }
}
