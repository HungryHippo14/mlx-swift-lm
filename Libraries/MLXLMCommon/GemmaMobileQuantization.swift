// Copyright 2026 Loci

import Foundation
import MLX
import MLXNN

#if GEMMA_MOBILE_QUANTIZATION_TESTS
import XCTest
#if GEMMA_MOBILE_STANDALONE
@testable import GemmaMobileIndependent
#else
@testable import MLXLMCommon
#endif

final class GemmaMobileQuantizationTests: XCTestCase {
    func testPackedWeightsRetainEverySignedValue() throws {
        for bits in [2, 4, 8] {
            let width = 128
            let offset = 1 << (bits - 1)
            let values = (0 ..< width).map { $0 % (1 << bits) - offset }
            let storage = packed(values, bits: bits, rows: 1, columns: width)
            let weights = try GemmaMobileWeights(
                packed: storage, scale: MLXArray([Float(0.125)], [1, 1]),
                rows: 1, columns: width, path: "test")
            XCTAssertEqual(weights.bits, bits)
            XCTAssertEqual(weights.weight.dtype, .uint32)
            XCTAssertEqual(weights.weight.nbytes, storage.nbytes)
            let decoded = dequantized(
                weights.weight, scales: weights.scales, biases: weights.biases,
                groupSize: weights.groupSize, bits: bits)
            XCTAssertEqual(decoded.asArray(Float.self), values.map { Float($0) * 0.125 })
        }
    }

    func testEmbeddingUsesPerBlockScalesAndOnlySelectedRows() throws {
        let width = 512
        let values = (0 ..< 3 * width).map { $0 % 16 - 8 }
        let weights = try GemmaMobileWeights(
            packed: packed(values, bits: 4, rows: 3, columns: width),
            scale: MLXArray([Float(1), 2, 3, 4, 5, 6], [3, 2]),
            rows: 3, columns: width, path: "embedding")
        let embedding = GemmaMobileEmbedding(weights: weights)
        let output = embedding(MLXArray([Int32(2), 0], [1, 2]))
        XCTAssertEqual(embedding.shape.0, 3)
        XCTAssertEqual(embedding.shape.1, width)
        XCTAssertEqual(output.shape, [1, 2, width])
        let expected = [2, 0].flatMap { row in
            (0 ..< width).map { column in
                Float(values[row * width + column]) * Float(row * 2 + column / 256 + 1)
            }
        }
        XCTAssertEqual(output.asArray(Float.self), expected)
    }

    func testStaticActivationRoundingClippingAndUncalibratedScale() {
        let input = MLXArray([Float(-100), -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 100])
        XCTAssertEqual(
            gemmaMobileSRQ(input, scale: MLXArray(Float(0.5))).asArray(Float.self),
            [-64, -1, -1, 0, 0, 1, 1, 63.5])
        XCTAssertEqual(
            gemmaMobileSRQ(input, scale: MLXArray(Float(0))).asArray(Float.self),
            input.asArray(Float.self))
    }

    func testFusedStaticRoundingMatchesEagerForBothRuntimeDtypes() {
        for dtype in [DType.float32, .bfloat16] {
            for scaleValue in [Float(0), 0.03, 0.5, 1.25] {
                let input = MLXArray(((-2048) ... 2048).map { Float($0) / 64 }).asType(dtype)
                let scale = MLXArray(scaleValue).asType(dtype)
                let calibrated = scale .!= 0
                let safeScale = MLX.where(calibrated, scale, MLXArray.ones(like: scale))
                let expected = MLX.where(
                    calibrated, clip(round(input / safeScale), min: -128, max: 127) * safeScale,
                    input)
                let actual = gemmaMobileSRQ(input, scale: scale)
                XCTAssertEqual(
                    max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self),
                    0, "dtype=\(dtype), scale=\(scaleValue)")
            }
        }
    }

    func testLinearMatchesDenseReferenceWithInputAndOutputRounding() throws {
        for bits in [2, 4, 8] {
            let width = 128
            let rows = 32
            let offset = 1 << (bits - 1)
            let values = (0 ..< rows * width).map { ($0 * 7 + $0 / width) % (1 << bits) - offset }
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: bits, rows: rows, columns: width),
                scale: MLXArray.full([rows, 1], values: MLXArray(Float(0.125))),
                rows: rows, columns: width, path: "linear")
            let inputScale = MLXArray(Float(0.5))
            let outputScale = MLXArray(Float(2))
            let bias = MLXArray((0 ..< rows).map { Float($0) * 0.25 })
            let linear = GemmaMobileLinear(
                weights: weights, bias: bias,
                inputScale: inputScale, outputScale: outputScale)
            let input = MLXArray((0 ..< 3 * width).map { Float($0 % 11 - 5) * 0.3 }, [3, width])
            let dense = MLXArray(values.map { Float($0) * 0.125 }, [rows, width])
            let expected = gemmaMobileSRQ(
                matmul(gemmaMobileSRQ(input, scale: inputScale), dense.T) + bias,
                scale: outputScale)
            XCTAssertEqual(linear(input).asArray(Float.self), expected.asArray(Float.self))
        }
    }

    func testBFloat16PackedKernelsMatchDenseReference() throws {
        for bits in [2, 4, 8] {
            let width = 128
            let rows = 32
            let offset = 1 << (bits - 1)
            let values = (0 ..< rows * width).map { ($0 * 7 + $0 / width) % (1 << bits) - offset }
            let scales = MLXArray.full([rows, 1], values: MLXArray(Float(0.03125))).asType(.bfloat16)
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: bits, rows: rows, columns: width),
                scale: scales, rows: rows, columns: width, path: "linear")
            let zero = MLXArray(Float(0)).asType(.bfloat16)
            let linear = GemmaMobileLinear(weights: weights, bias: nil, inputScale: zero, outputScale: zero)
            let input = MLXArray((0 ..< width).map { Float($0 % 7 - 3) * 0.125 }, [1, width]).asType(.bfloat16)
            let dense = MLXArray(values.map { Float($0) * 0.03125 }, [rows, width]).asType(.bfloat16)
            let actual = linear(input)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertEqual(actual.asType(.float32).asArray(Float.self), matmul(input, dense.T).asType(.float32).asArray(Float.self))
        }
    }

    func testBFloat16DequantizationMatchesSignedWeightScaleMultiplication() throws {
        for bits in [2, 4, 8] {
            let width = 512
            let rows = 4
            let offset = 1 << (bits - 1)
            let values = (0 ..< rows * width).map { $0 % (1 << bits) - offset }
            let originalScales = MLXArray(
                [Float(0.00194549560546875), 0.0054931640625, 0.0037689208984375,
                 0.004852294921875, 0.00732421875, 0.0021820068359375,
                 0.005401611328125, 0.0087890625],
                [rows, 2]).asType(.bfloat16)
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: bits, rows: rows, columns: width),
                scale: originalScales, rows: rows, columns: width, path: "embedding")
            let expected = MLXArray(values.map(Float.init), [rows, width]).asType(.bfloat16)
                * broadcast(originalScales.expandedDimensions(axis: -1), to: [rows, 2, 256])
                    .reshaped(rows, width)
            let actual = GemmaMobileEmbedding(weights: weights)(MLXArray([Int32(0), 1, 2, 3]))
            XCTAssertEqual(actual.asType(.float32).asArray(Float.self), expected.asType(.float32).asArray(Float.self))
        }
    }

    func testBFloat16LinearWithCheckpointScalesMatchesDenseReference() throws {
        let rowScales: [Float] = [
            0.00194549560546875, 0.0054931640625, 0.0037689208984375,
            0.004852294921875, 0.00732421875, 0.0021820068359375,
            0.005401611328125, 0.0087890625,
        ]
        for bits in [2, 4, 8] {
            let width = 512
            let rows = 32
            let offset = 1 << (bits - 1)
            let values = (0 ..< rows * width).map { ($0 * 7 + $0 / width) % (1 << bits) - offset }
            let scales = MLXArray((0 ..< rows).map { rowScales[$0 % rowScales.count] }, [rows, 1])
                .asType(.bfloat16)
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: bits, rows: rows, columns: width),
                scale: scales, rows: rows, columns: width, path: "linear")
            let zero = MLXArray(Float(0)).asType(.bfloat16)
            let linear = GemmaMobileLinear(weights: weights, bias: nil, inputScale: zero, outputScale: zero)
            let input = MLXArray((0 ..< 3 * width).map { Float(($0 * 13) % 73 - 36) / 37 }, [3, width])
                .asType(.bfloat16)
            let dense = MLXArray(values.map(Float.init), [rows, width]).asType(.bfloat16) * scales
            let expected = matmul(input, dense.T).asType(.float32)
            let actual = linear(input).asType(.float32)
            let error = actual - expected
            let maxError = max(abs(error)).item(Float.self)
            let relativeL2 = sqrt(sum(error * error) / sum(expected * expected)).item(Float.self)
            XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
            print("Gemma mobile checkpoint-scale linear bits=\(bits) max_abs_error=\(maxError) relative_l2_error=\(relativeL2)")
            XCTAssertLessThanOrEqual(relativeL2, 0.02)
        }
    }

    func testSmallBatchLinearAccumulatesBeforeBiasAndFinalRounding() throws {
        let width = 512
        let rows = 32
        for bits in [2, 4, 8] {
            let offset = 1 << (bits - 1)
            let values = (0 ..< rows * width).map { ($0 * 7 + $0 / width) % (1 << bits) - offset }
            let scales = MLXArray((0 ..< rows).map { Float($0 + 1) / 1024 }, [rows, 1])
                .asType(.bfloat16)
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: bits, rows: rows, columns: width),
                scale: scales, rows: rows, columns: width, path: "linear")
            let zero = MLXArray(Float(0)).asType(.bfloat16)
            let bias = MLXArray((0 ..< rows).map { Float($0 * 3 - 13) / 1024 }).asType(.bfloat16)
            let linear = GemmaMobileLinear(weights: weights, bias: bias, inputScale: zero, outputScale: zero)
            for batch in [1, 3, 31] {
                let input = MLXArray((0 ..< batch * width).map { Float(($0 * 13) % 73 - 36) / 37 }, [batch, width])
                    .asType(.bfloat16)
                let dense = MLXArray(values.map(Float.init), [rows, width]) * scales.asType(.float32)
                let expected = (matmul(input.asType(.float32), dense.T) + bias.asType(.float32))
                    .asType(.bfloat16)
                let actual = linear(input)
                XCTAssertEqual(actual.dtype, .bfloat16)
                XCTAssertEqual(actual.asType(.float32).asArray(Float.self), expected.asType(.float32).asArray(Float.self))
                let batched = linear(input.reshaped(1, batch, width))
                XCTAssertEqual(batched.shape, [1, batch, rows])
                XCTAssertEqual(batched.asType(.float32).asArray(Float.self), actual.asType(.float32).asArray(Float.self))
            }
        }
    }

    func testLargeBatchLinearKeepsBFloat16PrefillArithmetic() throws {
        let width = 512
        let rows = 32
        let values = (0 ..< rows * width).map { ($0 * 7 + $0 / width) % 16 - 8 }
        let scales = MLXArray((0 ..< rows).map { Float($0 + 1) / 1000 }, [rows, 1])
            .asType(.bfloat16)
        let weights = try GemmaMobileWeights(
            packed: packed(values, bits: 4, rows: rows, columns: width),
            scale: scales, rows: rows, columns: width, path: "linear")
        let zero = MLXArray(Float(0)).asType(.bfloat16)
        let linear = GemmaMobileLinear(weights: weights, bias: nil, inputScale: zero, outputScale: zero)
        for batch in [32, 64] {
            let input = MLXArray((0 ..< batch * width).map { Float(($0 * 13) % 73 - 36) / 37 }, [batch, width])
                .asType(.bfloat16)
            let dense = MLXArray(values.map(Float.init), [rows, width]).asType(.bfloat16) * scales
            let actual = linear(input)
            XCTAssertEqual(actual.dtype, .bfloat16)
            XCTAssertEqual(actual.asType(.float32).asArray(Float.self), matmul(input, dense.T).asType(.float32).asArray(Float.self))
            let batched = linear(input.reshaped(2, batch / 2, width))
            XCTAssertEqual(batched.shape, [2, batch / 2, rows])
            XCTAssertEqual(batched.asType(.float32).asArray(Float.self), actual.asType(.float32).asArray(Float.self))
        }
    }

    func testInvalidStorageAndScaleAreRejected() {
        for (storage, scales) in [
            (MLXArray.zeros([2, 32]), MLXArray.ones([2, 1])),
            (MLXArray.zeros([2, 32], type: UInt8.self), MLXArray.ones([3, 1])),
            (MLXArray.zeros([2, 32], type: UInt8.self), MLXArray.ones([2, 3])),
            (MLXArray.zeros([2, 32], type: UInt8.self), MLXArray.full([2, 1], values: MLXArray(Float.nan))),
            (MLXArray.zeros([2, 32], type: UInt8.self), MLXArray.full([2, 1], values: MLXArray(Float(-1)))),
        ] {
            XCTAssertThrowsError(try GemmaMobileWeights(
                packed: storage, scale: scales, rows: 2, columns: 128, path: "bad"))
        }
    }

    func testEmbeddingOutputProjectionAndGroupFallback() throws {
        for width in [32, 64, 128] {
            let values = (0 ..< 32 * width).map { $0 % 4 - 2 }
            let weights = try GemmaMobileWeights(
                packed: packed(values, bits: 2, rows: 32, columns: width),
                scale: MLXArray.ones([32, 1]), rows: 32, columns: width, path: "embedding")
            XCTAssertEqual(weights.groupSize, width)
            let embedding = GemmaMobileEmbedding(weights: weights)
            let input = MLXArray.ones([1, width])
            XCTAssertEqual(
                embedding.asLinear(input).asArray(Float.self),
                matmul(input, MLXArray(values.map(Float.init), [32, width]).T).asArray(Float.self))
        }
    }

    func testQuantizationDetectionLeavesOtherFormatsAlone() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertFalse(try usesGemmaMobileQuantization(modelDirectory: directory))
        for (contents, expected) in [
            (#"{"quantization_config":{"quant_method":"gemma"}}"#, true),
            (#"{"quantization":{"group_size":64,"bits":4}}"#, false),
            (#"{"quantization_config":{"quant_method":"compressed-tensors"}}"#, false),
        ] {
            try Data(contents.utf8).write(to: directory.appendingPathComponent("config.json"))
            XCTAssertEqual(try usesGemmaMobileQuantization(modelDirectory: directory), expected)
        }
    }

    func testLoaderReplacesModulesAndPreservesUnquantizedSiblings() throws {
        let model = TestModel()
        let linearValues = (0 ..< 32 * 128).map { $0 % 4 - 2 }
        let embeddingValues = (0 ..< 32 * 128).map { $0 % 16 - 8 }
        var weights: [String: MLXArray] = [
            "linear.weight": packed(linearValues, bits: 2, rows: 32, columns: 128),
            "linear.weight_scale": MLXArray.ones([32, 1]),
            "linear.input_activation_scale": MLXArray(Float(0.5)),
            "linear.output_activation_scale": MLXArray(Float(1)),
            "embedding.embedding_quantized": packed(embeddingValues, bits: 4, rows: 32, columns: 128),
            "embedding.embedding_scale": MLXArray.ones([32, 1]),
            "dense.weight": MLXArray.ones([32, 128]),
        ]
        weights = try applyGemmaMobileQuantization(model: model, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        XCTAssertTrue(model.linear is GemmaMobileLinear)
        XCTAssertEqual(weights["linear.scales"]?.dtype, .float32)
        XCTAssertEqual(weights["linear.biases"]?.dtype, .float32)
        XCTAssertTrue(model.embedding is GemmaMobileEmbedding)
        XCTAssertFalse(model.dense is Quantized)
        XCTAssertNil(weights["linear.weight_scale"])
        XCTAssertNil(weights["embedding.embedding_quantized"])
        XCTAssertEqual(model.linear(MLXArray.ones([1, 128])).shape, [1, 32])
    }

    func testUnexpectedQuantizedModuleIsRejectedBeforeMutation() {
        let model = TestModel()
        XCTAssertThrowsError(try applyGemmaMobileQuantization(model: model, weights: [
            "missing.weight_scale": MLXArray.ones([1, 1]),
            "missing.weight": MLXArray.zeros([1, 32], type: UInt8.self),
        ]))
        XCTAssertFalse(model.linear is Quantized)
    }

    private func packed(_ values: [Int], bits: Int, rows: Int, columns: Int) -> MLXArray {
        if bits == 8 {
            return MLXArray(values.map { Int8($0) }, [rows, columns])
        }
        let offset = 1 << (bits - 1)
        let perByte = 8 / bits
        let bytes = stride(from: 0, to: values.count, by: perByte).map { start in
            (0 ..< perByte).reduce(UInt8(0)) { result, index in
                result | UInt8(values[start + index] + offset) << (bits * index)
            }
        }
        return MLXArray(bytes, [rows, columns / perByte])
    }

    private final class TestModel: Module {
        @ModuleInfo var linear = Linear(128, 32, bias: false)
        @ModuleInfo var embedding = Embedding(embeddingCount: 32, dimensions: 128)
        @ModuleInfo var dense = Linear(128, 32, bias: false)
    }
}
#else

enum GemmaMobileQuantizationError: Error, LocalizedError {
    case invalidTensor(String)
    case unsupportedModule(String)
    case missingTensor(String)

    var errorDescription: String? {
        switch self {
        case .invalidTensor(let path): "Invalid Gemma mobile quantization tensor: \(path)"
        case .unsupportedModule(let path): "Unsupported Gemma mobile quantization module: \(path)"
        case .missingTensor(let path): "Missing Gemma mobile quantization tensor: \(path)"
        }
    }
}

struct GemmaMobileWeights {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let bits: Int
    let groupSize: Int

    init(packed: MLXArray, scale: MLXArray, rows: Int, columns: Int, path: String) throws {
        guard packed.ndim == 2, packed.dim(0) == rows, packed.dim(1) > 0,
            rows > 0, columns > 0, packed.dim(1) % 4 == 0,
            scale.ndim == 2, scale.dim(0) == rows, scale.dim(1) > 0,
            scale.dtype == .float32 || scale.dtype == .float16 || scale.dtype == .bfloat16,
            columns % scale.dim(1) == 0,
            let groupSize = [128, 64, 32].first(where: {
                columns % $0 == 0 && (columns / scale.dim(1)) % $0 == 0
            })
        else { throw GemmaMobileQuantizationError.invalidTensor(path) }
        let bits = packed.dim(1) * 8 / columns
        guard [2, 4, 8].contains(bits), packed.dim(1) * 8 == columns * bits,
            packed.dtype == (bits == 8 ? .int8 : .uint8),
            all(isFinite(scale)).item(Bool.self), all(scale .>= 0).item(Bool.self)
        else { throw GemmaMobileQuantizationError.invalidTensor(path) }
        self.bits = bits
        self.groupSize = groupSize
        let unsigned = bits == 8
            ? bitwiseXOr(packed.view(dtype: .uint8), MLXArray(UInt8(128))) : packed
        self.weight = unsigned.view(dtype: .uint32)
        let repeats = columns / groupSize / scale.dim(1)
        self.scales = broadcast(
            scale.expandedDimensions(axis: -1),
            to: [rows, scale.dim(1), repeats]).reshaped(rows, columns / groupSize)
        self.biases = self.scales * -Float(1 << (bits - 1))
    }
}

private let gemmaMobileCompiledSRQ: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { input, scale in
    let scale = scale.asType(input.dtype)
    let calibrated = scale .!= 0
    let safeScale = MLX.where(calibrated, scale, MLXArray.ones(like: scale))
    let quantized = clip(round(input / safeScale), min: -128, max: 127) * safeScale
    return MLX.where(calibrated, quantized, input)
}

func gemmaMobileSRQ(_ input: MLXArray, scale: MLXArray) -> MLXArray {
    gemmaMobileCompiledSRQ(input, scale)
}

final class GemmaMobileLinear: QuantizedLinear {
    @ParameterInfo(key: "input_activation_scale") var inputActivationScale: MLXArray
    @ParameterInfo(key: "output_activation_scale") var outputActivationScale: MLXArray

    init(weights: GemmaMobileWeights, bias: MLXArray?, inputScale: MLXArray, outputScale: MLXArray) {
        self._inputActivationScale.wrappedValue = inputScale
        self._outputActivationScale.wrappedValue = outputScale
        super.init(
            weight: weights.weight, bias: bias,
            scales: weights.scales.asType(.float32), biases: weights.biases.asType(.float32),
            groupSize: weights.groupSize, bits: weights.bits)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let roundedInput = gemmaMobileSRQ(input, scale: inputActivationScale)
        let accumulationType = input.size / input.dim(-1) < 32 ? DType.float32 : input.dtype
        var output = quantizedMM(
            roundedInput.asType(accumulationType), weight,
            scales: scales.asType(accumulationType), biases: biases?.asType(accumulationType),
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        if let bias {
            output = output + bias.asType(accumulationType)
        }
        return gemmaMobileSRQ(output.asType(input.dtype), scale: outputActivationScale)
    }
}

final class GemmaMobileEmbedding: Embedding, Quantized {
    let scales: MLXArray
    let biases: MLXArray
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode = .affine

    override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 32 / bits) }

    init(weights: GemmaMobileWeights) {
        self.scales = weights.scales
        self.biases = weights.biases
        self.groupSize = weights.groupSize
        self.bits = weights.bits
        super.init(weight: weights.weight)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let indices = input.flattened()
        return dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices],
            groupSize: groupSize, bits: bits).reshaped(input.shape + [-1])
    }

    override func asLinear(_ input: MLXArray) -> MLXArray {
        quantizedMM(
            input, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}

func applyGemmaMobileQuantization(
    model: Module, weights: [String: MLXArray]
) throws -> [String: MLXArray] {
    let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
    var transformed = weights
    var updates: [(String, Module)] = []
    let scaleKeys = weights.keys.filter {
        $0.hasSuffix(".weight_scale") || $0.hasSuffix(".embedding_scale")
    }.sorted()
    guard !scaleKeys.isEmpty else {
        throw GemmaMobileQuantizationError.missingTensor("weight_scale")
    }
    for scaleKey in scaleKeys {
        let isEmbedding = scaleKey.hasSuffix(".embedding_scale")
        let suffix = isEmbedding ? ".embedding_scale" : ".weight_scale"
        let path = String(scaleKey.dropLast(suffix.count))
        let weightKey = path + (isEmbedding ? ".embedding_quantized" : ".weight")
        guard let module = modules[path] else {
            throw GemmaMobileQuantizationError.unsupportedModule(path)
        }
        guard let packed = weights[weightKey], let scale = weights[scaleKey] else {
            throw GemmaMobileQuantizationError.missingTensor(weightKey)
        }
        let dimensions: (Int, Int)
        if isEmbedding, let embedding = module as? Embedding {
            dimensions = embedding.shape
        } else if !isEmbedding, let linear = module as? Linear {
            dimensions = linear.shape
            guard (linear.bias != nil) == (weights[path + ".bias"] != nil) else {
                throw GemmaMobileQuantizationError.missingTensor(path + ".bias")
            }
            if let bias = weights[path + ".bias"], bias.shape != [dimensions.0] {
                throw GemmaMobileQuantizationError.invalidTensor(path + ".bias")
            }
        } else {
            throw GemmaMobileQuantizationError.unsupportedModule(path)
        }
        let converted = try GemmaMobileWeights(
            packed: packed, scale: scale, rows: dimensions.0, columns: dimensions.1, path: path)
        let replacement: Module
        if isEmbedding {
            replacement = GemmaMobileEmbedding(weights: converted)
        } else {
            let inputKey = path + ".input_activation_scale"
            let outputKey = path + ".output_activation_scale"
            let inputScale = weights[inputKey] ?? MLXArray(Float(0))
            let outputScale = weights[outputKey] ?? MLXArray(Float(0))
            guard inputScale.ndim == 0, outputScale.ndim == 0,
                inputScale.dtype.isFloatingPoint, outputScale.dtype.isFloatingPoint,
                all(isFinite(inputScale)).item(Bool.self), all(inputScale .>= 0).item(Bool.self),
                all(isFinite(outputScale)).item(Bool.self), all(outputScale .>= 0).item(Bool.self)
            else {
                throw GemmaMobileQuantizationError.invalidTensor(path + ".activation_scale")
            }
            transformed[inputKey] = inputScale
            transformed[outputKey] = outputScale
            replacement = GemmaMobileLinear(
                weights: converted, bias: weights[path + ".bias"],
                inputScale: inputScale, outputScale: outputScale)
        }
        transformed.removeValue(forKey: scaleKey)
        transformed.removeValue(forKey: weightKey)
        transformed[path + ".weight"] = converted.weight
        if let linear = replacement as? GemmaMobileLinear {
            transformed[path + ".scales"] = linear.scales
            transformed[path + ".biases"] = linear.biases
        } else {
            transformed[path + ".scales"] = converted.scales
            transformed[path + ".biases"] = converted.biases
        }
        updates.append((path, replacement))
    }
    model.update(modules: ModuleChildren.unflattened(updates))
    return transformed
}

func usesGemmaMobileQuantization(modelDirectory: URL) throws -> Bool {
    struct Configuration: Decodable {
        struct Quantization: Decodable {
            let quantMethod: String?
            enum CodingKeys: String, CodingKey { case quantMethod = "quant_method" }
        }
        let quantization: Quantization?
        enum CodingKeys: String, CodingKey { case quantization = "quantization_config" }
    }
    let path = modelDirectory.appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: path.path) else { return false }
    let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: path))
    return config.quantization?.quantMethod == "gemma"
}
#endif
