package run.endive.simd;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.function.LongBinaryOperator;

/** Test-only scalar reference for the 19 instructions exercised by {@link SimdDifferentialTest}. */
final class ScalarV128Reference {

    static final int BYTE_SIZE = 16;

    private ScalarV128Reference() {}

    static byte[] constant(byte[] value) {
        requireVector(value);
        return value.clone();
    }

    static byte[] load(byte[] memory, int offset) {
        requireRange(memory, offset);
        return Arrays.copyOfRange(memory, offset, offset + BYTE_SIZE);
    }

    static void store(byte[] memory, int offset, byte[] value) {
        requireVector(value);
        requireRange(memory, offset);
        System.arraycopy(value, 0, memory, offset, BYTE_SIZE);
    }

    static byte[] and(byte[] left, byte[] right) {
        return bitwise(left, right, (a, b) -> a & b);
    }

    static byte[] or(byte[] left, byte[] right) {
        return bitwise(left, right, (a, b) -> a | b);
    }

    static byte[] xor(byte[] left, byte[] right) {
        return bitwise(left, right, (a, b) -> a ^ b);
    }

    static byte[] not(byte[] value) {
        requireVector(value);
        var result = new byte[BYTE_SIZE];
        for (var i = 0; i < BYTE_SIZE; i++) {
            result[i] = (byte) ~value[i];
        }
        return result;
    }

    static byte[] i8x16Add(byte[] left, byte[] right) {
        return binaryLanes(left, right, 1, (a, b) -> a + b);
    }

    static byte[] i8x16Sub(byte[] left, byte[] right) {
        return binaryLanes(left, right, 1, (a, b) -> a - b);
    }

    static byte[] i16x8Add(byte[] left, byte[] right) {
        return binaryLanes(left, right, 2, (a, b) -> a + b);
    }

    static byte[] i32x4Add(byte[] left, byte[] right) {
        return binaryLanes(left, right, 4, (a, b) -> a + b);
    }

    static byte[] i32x4Sub(byte[] left, byte[] right) {
        return binaryLanes(left, right, 4, (a, b) -> a - b);
    }

    static byte[] i32x4Mul(byte[] left, byte[] right) {
        return binaryLanes(left, right, 4, (a, b) -> a * b);
    }

    static byte[] i64x2Add(byte[] left, byte[] right) {
        return binaryLanes(left, right, 8, (a, b) -> a + b);
    }

    static byte[] i64x2Sub(byte[] left, byte[] right) {
        return binaryLanes(left, right, 8, (a, b) -> a - b);
    }

    static byte[] i64x2Shl(byte[] value, int shift) {
        requireVector(value);
        var input = wrap(value);
        var result = allocateVector();
        for (var lane = 0; lane < 2; lane++) {
            result.putLong(input.getLong() << (shift & 63));
        }
        return result.array();
    }

    static byte[] i64x2ShrU(byte[] value, int shift) {
        requireVector(value);
        var input = wrap(value);
        var result = allocateVector();
        for (var lane = 0; lane < 2; lane++) {
            result.putLong(input.getLong() >>> (shift & 63));
        }
        return result.array();
    }

    static int i32x4ExtractLane(byte[] value, int lane) {
        requireVector(value);
        if (lane < 0 || lane >= 4) {
            throw new IllegalArgumentException("Invalid i32x4 lane: " + lane);
        }
        return wrap(value).getInt(lane * Integer.BYTES);
    }

    static byte[] i8x16Shuffle(byte[] left, byte[] right, int[] lanes) {
        requireVector(left);
        requireVector(right);
        if (lanes.length != BYTE_SIZE) {
            throw new IllegalArgumentException("Expected 16 shuffle lanes, got " + lanes.length);
        }

        var result = new byte[BYTE_SIZE];
        for (var lane = 0; lane < BYTE_SIZE; lane++) {
            var source = lanes[lane];
            if (source < 0 || source >= BYTE_SIZE * 2) {
                throw new IllegalArgumentException("Invalid shuffle lane: " + source);
            }
            result[lane] = source < BYTE_SIZE ? left[source] : right[source - BYTE_SIZE];
        }
        return result;
    }

    private static byte[] bitwise(byte[] left, byte[] right, LongBinaryOperator operation) {
        requireVector(left);
        requireVector(right);
        var result = new byte[BYTE_SIZE];
        for (var i = 0; i < BYTE_SIZE; i++) {
            result[i] = (byte) operation.applyAsLong(left[i], right[i]);
        }
        return result;
    }

    private static byte[] binaryLanes(
            byte[] left, byte[] right, int laneBytes, LongBinaryOperator operation) {
        requireVector(left);
        requireVector(right);
        var leftLanes = wrap(left);
        var rightLanes = wrap(right);
        var result = allocateVector();
        for (var lane = 0; lane < BYTE_SIZE / laneBytes; lane++) {
            switch (laneBytes) {
                case 1:
                    result.put((byte) operation.applyAsLong(leftLanes.get(), rightLanes.get()));
                    break;
                case 2:
                    result.putShort(
                            (short)
                                    operation.applyAsLong(
                                            leftLanes.getShort(), rightLanes.getShort()));
                    break;
                case 4:
                    result.putInt(
                            (int) operation.applyAsLong(leftLanes.getInt(), rightLanes.getInt()));
                    break;
                case 8:
                    result.putLong(
                            operation.applyAsLong(leftLanes.getLong(), rightLanes.getLong()));
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported lane width: " + laneBytes);
            }
        }
        return result.array();
    }

    private static ByteBuffer wrap(byte[] value) {
        return ByteBuffer.wrap(value).order(LITTLE_ENDIAN);
    }

    private static ByteBuffer allocateVector() {
        return ByteBuffer.allocate(BYTE_SIZE).order(LITTLE_ENDIAN);
    }

    private static void requireVector(byte[] value) {
        if (value.length != BYTE_SIZE) {
            throw new IllegalArgumentException("Expected a 16-byte vector, got " + value.length);
        }
    }

    private static void requireRange(byte[] memory, int offset) {
        if (offset < 0 || offset > memory.length - BYTE_SIZE) {
            throw new IndexOutOfBoundsException(
                    "A 16-byte vector does not fit at offset "
                            + offset
                            + " in "
                            + memory.length
                            + " bytes");
        }
    }
}
