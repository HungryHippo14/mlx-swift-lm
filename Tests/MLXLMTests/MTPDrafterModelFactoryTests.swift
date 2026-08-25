// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

private final class QuantizableMockMTPDrafter: Module, MTPDrafterModel {
    @ModuleInfo(key: "projection") var projection: Linear

    override init() {
        _projection.wrappedValue = Linear(64, 64, bias: false)
    }

    func draftBlock(
        target _: any LanguageModel,
        lastToken _: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize _: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        fatalError("not used by the factory-load regression")
    }
}

private struct UnusedTokenizerLoader: TokenizerLoader {
    func load(from _: URL) async throws -> any Tokenizer {
        fatalError("MTPDrafterModelFactory must not load a tokenizer")
    }
}

// MARK: - Type-registry registration

@Test
func testGemma4AssistantRegistrationRegistersType() async throws {
    await Gemma4AssistantRegistration.register()

    let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 4,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "global_head_dim": 2,
            "vocab_size": 10,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 4,
            "sliding_window_pattern": 1,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_traditional": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "attention_k_eq_v": true,
            "intermediate_size": 8,
            "layer_types": ["full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let data = Data(json.utf8)

    let model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: data, modelType: "gemma4_assistant")
    #expect(model is Gemma4AssistantDraftModel)
}

@Test
func testMTPDrafterTypeRegistryUnknownModelTypeThrows() async {
    // Don't pre-register; an unknown model type must throw
    // `unsupportedModelType`. Use a distinctive name so it doesn't collide
    // with any registration another test made (registration is shared).
    do {
        let _: any MTPDrafterModel =
            try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: Data(),
                modelType: "definitely_not_a_drafter_type")
        Issue.record("expected unsupportedModelType to throw")
    } catch let error as ModelFactoryError {
        if case .unsupportedModelType(let t) = error {
            #expect(t == "definitely_not_a_drafter_type")
        } else {
            Issue.record("unexpected ModelFactoryError: \(error)")
        }
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

@Test("Standalone MTP factory applies top-level quantization")
func testStandaloneMTPFactoryAppliesGlobalQuantization() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appending(component: "mtp-global-quantization-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let reference = QuantizableMockMTPDrafter()
    let (weight, scales, biases) = quantized(reference.projection.weight, groupSize: 64, bits: 4)
    var checkpoint = [
        "projection.weight": weight,
        "projection.scales": scales,
    ]
    if let biases {
        checkpoint["projection.biases"] = biases
    }
    try save(arrays: checkpoint, url: directory.appending(component: "model.safetensors"))
    try Data(
        """
        {
          "model_type": "quantizable_mock_mtp",
          "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
        }
        """.utf8
    ).write(to: directory.appending(component: "config.json"))

    let typeRegistry = ModelTypeRegistry<any MTPDrafterModel>(
        creators: ["quantizable_mock_mtp": { _ in QuantizableMockMTPDrafter() }])
    let factory = MTPDrafterModelFactory(
        typeRegistry: typeRegistry, modelRegistry: AbstractModelRegistry())

    let context = try await factory._load(
        configuration: .init(directory: directory),
        tokenizerLoader: UnusedTokenizerLoader())
    let loaded = try #require(context.model as? QuantizableMockMTPDrafter)
    let projection = try #require(loaded.projection as? QuantizedLinear)
    #expect(projection.groupSize == 64)
    #expect(projection.bits == 4)
}

// MARK: - Model registry contents

@Test
func testMTPDrafterRegistryContainsBothReferenceCheckpoints() {
    let registry = MTPDrafterRegistry.shared
    #expect(registry.contains(id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"))
    #expect(registry.contains(id: "mlx-community/gemma-4-31B-it-assistant-bf16"))
}

@Test
func testMTPDrafterRegistrySharedStaticAccessors() {
    let r26 = MTPDrafterRegistry.gemma4_26B_assistant_bf16
    let r31 = MTPDrafterRegistry.gemma4_31B_assistant_bf16
    // Don't depend on a specific case enumeration shape; just verify that
    // the registered name property contains the expected substring.
    #expect(r26.name.contains("26B-A4B-it-assistant"))
    #expect(r31.name.contains("31B-it-assistant"))
}
