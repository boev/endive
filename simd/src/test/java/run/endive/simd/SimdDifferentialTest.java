package run.endive.simd;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import run.endive.runtime.Instance;
import run.endive.wabt.Wat2Wasm;
import run.endive.wasm.Parser;

/**
 * Compares the Vector API interpreter with a scalar reference for 19 SIMD instructions using edge
 * values and seeded random inputs. This is test-only groundwork for #181.
 */
// tests share one instance and write its memory
@Execution(ExecutionMode.SAME_THREAD)
public class SimdDifferentialTest {

    private static final long RANDOM_SEED = 42L;
    private static final int RANDOM_CASES = 32;
    private static final int MEMORY_OFFSET = 3; // deliberately unaligned for v128 loads and stores
    private static final int MEMORY_SIZE = 32;
    private static final byte MEMORY_FILL = 0x5a;
    // lanes must match the i8x16.shuffle immediate in the WAT fixture
    private static final int[] SHUFFLE_LANES = {
        0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31
    };
    // bytes must match the v128.const literal in the WAT fixture
    private static final byte[] CONSTANT = {
        0, -1, Byte.MIN_VALUE, Byte.MAX_VALUE, 1, -2, 126, -127,
        0, -1, Byte.MIN_VALUE, Byte.MAX_VALUE, 1, -2, 126, -127
    };
    private static final Instance SIMD = createInstance();

    @Test
    public void shouldMatchConstantReference() {
        assertVectorEquals(
                "v128.const",
                List.of(formatVector(CONSTANT)),
                ScalarV128Reference.constant(CONSTANT),
                invokeVector("v128.const"));
    }

    @Test
    public void shouldMatchLoadReference() {
        for (var input : inputs("v128.load", 1)) {
            var memory = memoryContaining(input);
            SIMD.memory(0).write(0, memory);

            assertVectorEquals(
                    "v128.load",
                    List.of(formatVector(memory), Integer.toString(MEMORY_OFFSET)),
                    ScalarV128Reference.load(memory, MEMORY_OFFSET),
                    invokeVector("v128.load", MEMORY_OFFSET));
        }
    }

    @Test
    public void shouldMatchStoreReference() {
        for (var input : inputs("v128.store", 1)) {
            var initialMemory = new byte[MEMORY_SIZE];
            Arrays.fill(initialMemory, MEMORY_FILL);
            var expectedMemory = initialMemory.clone();
            ScalarV128Reference.store(expectedMemory, MEMORY_OFFSET, input);

            SIMD.memory(0).write(0, initialMemory);
            SIMD.export("v128.store").apply(withAddress(MEMORY_OFFSET, input));
            var actualMemory = SIMD.memory(0).readBytes(0, initialMemory.length);

            assertVectorEquals(
                    "v128.store",
                    List.of(
                            formatVector(initialMemory),
                            Integer.toString(MEMORY_OFFSET),
                            formatVector(input)),
                    expectedMemory,
                    actualMemory);
        }
    }

    @Test
    public void shouldMatchBitwiseReference() {
        checkBinary("v128.and", 1, ScalarV128Reference::and);
        checkBinary("v128.or", 1, ScalarV128Reference::or);
        checkBinary("v128.xor", 1, ScalarV128Reference::xor);
        checkUnary("v128.not", 1, ScalarV128Reference::not);
    }

    @Test
    public void shouldMatchLaneArithmeticReference() {
        checkBinary("i8x16.add", 1, ScalarV128Reference::i8x16Add);
        checkBinary("i8x16.sub", 1, ScalarV128Reference::i8x16Sub);
        checkBinary("i16x8.add", 2, ScalarV128Reference::i16x8Add);
        checkBinary("i32x4.add", 4, ScalarV128Reference::i32x4Add);
        checkBinary("i32x4.sub", 4, ScalarV128Reference::i32x4Sub);
        checkBinary("i32x4.mul", 4, ScalarV128Reference::i32x4Mul);
        checkBinary("i64x2.add", 8, ScalarV128Reference::i64x2Add);
        checkBinary("i64x2.sub", 8, ScalarV128Reference::i64x2Sub);
    }

    @Test
    public void shouldMatchShiftReference() {
        checkShift("i64x2.shl", ScalarV128Reference::i64x2Shl);
        checkShift("i64x2.shr_u", ScalarV128Reference::i64x2ShrU);
    }

    @Test
    public void shouldMatchExtractLaneReference() {
        for (var input : inputs("i32x4.extract_lane", 4)) {
            for (var lane = 0; lane < 4; lane++) {
                var expected = (long) ScalarV128Reference.i32x4ExtractLane(input, lane);
                var actual = SIMD.export("i32x4.extract_lane." + lane).apply(toLongs(input))[0];
                assertEquals(
                        expected,
                        actual,
                        failureMessage(
                                "i32x4.extract_lane " + lane,
                                List.of(formatVector(input)),
                                Long.toString(expected),
                                Long.toString(actual)));
            }
        }
    }

    @Test
    public void shouldMatchShuffleReference() {
        checkBinary(
                "i8x16.shuffle",
                1,
                (left, right) -> ScalarV128Reference.i8x16Shuffle(left, right, SHUFFLE_LANES));
    }

    private static void checkUnary(
            String opcode, int laneBytes, Function<byte[], byte[]> reference) {
        for (var input : inputs(opcode, laneBytes)) {
            assertVectorEquals(
                    opcode,
                    List.of(formatVector(input)),
                    reference.apply(input),
                    invokeVector(opcode, toLongs(input)));
        }
    }

    private static void checkBinary(
            String opcode, int laneBytes, BiFunction<byte[], byte[], byte[]> reference) {
        var inputs = inputs(opcode, laneBytes);
        for (var i = 0; i < inputs.size(); i++) {
            var left = inputs.get(i);
            // reverse pairing exercises every case in both operand positions
            var right = inputs.get(inputs.size() - i - 1);
            assertVectorEquals(
                    opcode,
                    List.of(formatVector(left), formatVector(right)),
                    reference.apply(left, right),
                    invokeVector(opcode, concatenate(toLongs(left), toLongs(right))));
        }
    }

    private static void checkShift(String opcode, BiFunction<byte[], Integer, byte[]> reference) {
        var shifts = new int[] {0, 1, 63, 64, 65, -1, Integer.MIN_VALUE, Integer.MAX_VALUE};
        for (var input : inputs(opcode, 8)) {
            for (var shift : shifts) {
                assertVectorEquals(
                        opcode,
                        List.of(formatVector(input), Integer.toString(shift)),
                        reference.apply(input, shift),
                        invokeVector(opcode, withShift(input, shift)));
            }
        }
    }

    private static List<byte[]> inputs(String opcode, int laneBytes) {
        var inputs = edgeInputs(laneBytes);
        // each opcode gets reproducible random inputs, independent of test execution order
        var random = new Random(RANDOM_SEED ^ opcode.hashCode());
        for (var i = 0; i < RANDOM_CASES; i++) {
            var input = new byte[ScalarV128Reference.BYTE_SIZE];
            random.nextBytes(input);
            inputs.add(input);
        }
        return inputs;
    }

    private static List<byte[]> edgeInputs(int laneBytes) {
        var min = 0L;
        var max = 0L;
        switch (laneBytes) {
            case 1:
                min = Byte.MIN_VALUE;
                max = Byte.MAX_VALUE;
                break;
            case 2:
                min = Short.MIN_VALUE;
                max = Short.MAX_VALUE;
                break;
            case 4:
                min = Integer.MIN_VALUE;
                max = Integer.MAX_VALUE;
                break;
            case 8:
                min = Long.MIN_VALUE;
                max = Long.MAX_VALUE;
                break;
            default:
                throw new IllegalArgumentException("Unsupported lane width: " + laneBytes);
        }

        var edgeValues = new long[] {0, -1, min, max, min + 1, max - 1, 1, -2};
        var inputs = new ArrayList<byte[]>();
        for (var rotation = 0; rotation < edgeValues.length; rotation++) {
            var buffer = littleEndianVector();
            for (var lane = 0; lane < ScalarV128Reference.BYTE_SIZE / laneBytes; lane++) {
                putLane(buffer, laneBytes, edgeValues[(lane + rotation) % edgeValues.length]);
            }
            inputs.add(buffer.array());
        }
        return inputs;
    }

    private static void putLane(ByteBuffer buffer, int laneBytes, long value) {
        for (var i = 0; i < laneBytes; i++) {
            buffer.put((byte) (value >>> (i * Byte.SIZE)));
        }
    }

    private static byte[] memoryContaining(byte[] input) {
        var memory = new byte[MEMORY_SIZE];
        Arrays.fill(memory, MEMORY_FILL);
        System.arraycopy(input, 0, memory, MEMORY_OFFSET, input.length);
        return memory;
    }

    private static long[] toLongs(byte[] value) {
        if (value.length != ScalarV128Reference.BYTE_SIZE) {
            throw new IllegalArgumentException("Expected a 16-byte vector, got " + value.length);
        }
        var buffer = ByteBuffer.wrap(value).order(LITTLE_ENDIAN);
        return new long[] {buffer.getLong(), buffer.getLong()};
    }

    private static byte[] fromLongs(long[] value) {
        if (value.length != 2) {
            throw new IllegalArgumentException(
                    "Expected two v128 stack slots, got " + value.length);
        }
        var buffer = littleEndianVector();
        buffer.putLong(value[0]);
        buffer.putLong(value[1]);
        return buffer.array();
    }

    private static long[] withAddress(int address, byte[] value) {
        var vector = toLongs(value);
        return new long[] {address, vector[0], vector[1]};
    }

    private static long[] withShift(byte[] value, int shift) {
        var vector = toLongs(value);
        return new long[] {vector[0], vector[1], shift};
    }

    private static long[] concatenate(long[] left, long[] right) {
        return new long[] {left[0], left[1], right[0], right[1]};
    }

    private static ByteBuffer littleEndianVector() {
        return ByteBuffer.allocate(ScalarV128Reference.BYTE_SIZE).order(LITTLE_ENDIAN);
    }

    private static byte[] invokeVector(String export, long... arguments) {
        return fromLongs(SIMD.export(export).apply(arguments));
    }

    private static void assertVectorEquals(
            String opcode, List<String> inputs, byte[] expected, byte[] actual) {
        assertArrayEquals(
                expected,
                actual,
                () -> failureMessage(opcode, inputs, formatVector(expected), formatVector(actual)));
    }

    private static String failureMessage(
            String opcode, List<String> inputs, String expected, String actual) {
        var message = "opcode=" + opcode;
        for (var i = 0; i < inputs.size(); i++) {
            message += "\ninput[" + i + "]=" + inputs.get(i);
        }
        return message + "\nexpected(scalar)=" + expected + "\nactual(vector)=" + actual;
    }

    private static String formatVector(byte[] value) {
        var formatted = "bytes(le)=[";
        for (var i = 0; i < value.length; i++) {
            if (i > 0) {
                formatted += " ";
            }
            formatted += String.format("%02x", Byte.toUnsignedInt(value[i]));
        }
        return formatted + "]";
    }

    private static Instance createInstance() {
        try (var input = SimdDifferentialTest.class.getResourceAsStream("/simd-differential.wat")) {
            if (input == null) {
                throw new IllegalStateException("Missing test resource: /simd-differential.wat");
            }
            var wat = new String(input.readAllBytes(), UTF_8);
            return Instance.builder(Parser.parse(Wat2Wasm.parse(wat)))
                    .withMachineFactory(SimdInterpreterMachine::new)
                    .build();
        } catch (IOException e) {
            throw new IllegalStateException("Could not read /simd-differential.wat", e);
        }
    }
}
