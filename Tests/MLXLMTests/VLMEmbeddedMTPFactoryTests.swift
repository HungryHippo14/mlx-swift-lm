// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class VLMEmbeddedMTPFactoryTests: XCTestCase {

    func testFullRawQwenCheckpointReturnsTargetAndEmbeddedDrafter() async throws {
        try await assertFullCheckpointLoad(metadata: [:], expectedNormValue: 1)
    }

    func testMLXMetadataCheckpointReturnsTargetAndEmbeddedDrafterWithoutDoubleShift() async throws {
        try await assertFullCheckpointLoad(
            metadata: ["format": "mlx"], expectedNormValue: 0)
    }

    func testEmbeddedMTPAPIRejectsUnsupportedModelTypeBeforeWeightLoad() async throws {
        let configurationData = qwenVLMConfigurationData(modelType: "unsupported_vlm")
        let directory = try makeDirectory(configurationData: configurationData)
        defer { try? FileManager.default.removeItem(at: directory) }
        let typeRegistry = ModelTypeRegistry<LanguageModel>(creators: [
            "unsupported_vlm": { _ in
                XCTFail("An unsupported embedded-MTP type must fail before model construction")
                return NonQwenVLMTarget()
            }
        ])
        let factory = makeFactory(typeRegistry: typeRegistry)

        do {
            _ = try await factory.loadQwenWithEmbeddedMTP(
                from: directory, using: EmbeddedMTPTestTokenizerLoader())
            XCTFail("Expected an unsupported model type to fail closed")
        } catch ModelFactoryError.unsupportedModelType(let modelType) {
            XCTAssertEqual(modelType, "unsupported_vlm")
        }
    }

    func testEmbeddedMTPAPIRejectsNonQwenRegistryTargetBeforeWeightLoad() async throws {
        let directory = try makeDirectory(configurationData: qwenVLMConfigurationData())
        defer { try? FileManager.default.removeItem(at: directory) }
        let typeRegistry = ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_5": { _ in NonQwenVLMTarget() }
        ])
        let factory = makeFactory(typeRegistry: typeRegistry)

        do {
            _ = try await factory.loadQwenWithEmbeddedMTP(
                from: directory, using: EmbeddedMTPTestTokenizerLoader())
            XCTFail("Expected a non-Qwen target to fail closed")
        } catch ModelFactoryError.invalidConfiguration(let message) {
            XCTAssertTrue(message.contains("incompatible with target"))
            XCTAssertTrue(message.contains("NonQwenVLMTarget"))
        }
    }

    func testEmbeddedMTPAPIRejectsMismatchedQwenModelTypeAndClass() async throws {
        let configurationData = qwenVLMConfigurationData(modelType: "qwen3_5_moe")
        let directory = try makeDirectory(configurationData: configurationData)
        defer { try? FileManager.default.removeItem(at: directory) }
        let typeRegistry = ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_5_moe": { data in
                Qwen35(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data))
            }
        ])
        let factory = makeFactory(typeRegistry: typeRegistry)

        do {
            _ = try await factory.loadQwenWithEmbeddedMTP(
                from: directory, using: EmbeddedMTPTestTokenizerLoader())
            XCTFail("Expected an exact model-type/class mismatch to fail closed")
        } catch ModelFactoryError.invalidConfiguration(let message) {
            XCTAssertTrue(message.contains("qwen3_5_moe"))
            XCTAssertTrue(message.contains("incompatible with target"))
        }
    }

    private func assertFullCheckpointLoad(
        metadata: [String: String],
        expectedNormValue: Float
    ) async throws {
        let configurationData = qwenVLMConfigurationData()
        let directory = try makeDirectory(configurationData: configurationData)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: configurationData)
        let target = Qwen35(configuration)
        let drafter = Qwen35VLMNextNDraftModel(configuration)
        var checkpoint = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        for (key, value) in drafter.parameters().flattened() {
            XCTAssertNil(checkpoint.updateValue(value, forKey: key))
        }
        checkpoint["language_model.model.norm.weight"] = MLXArray.zeros([16])
        checkpoint["mtp.norm.weight"] = MLXArray.zeros([16])
        let weightsURL = directory.appending(component: "model.safetensors")
        if metadata.isEmpty {
            try save(arrays: checkpoint, url: weightsURL)
        } else {
            try save(arrays: checkpoint, metadata: metadata, url: weightsURL)
        }

        let loaded = try await makeFactory().loadQwenWithEmbeddedMTP(
            from: directory, using: EmbeddedMTPTestTokenizerLoader())
        let loadedTarget = try XCTUnwrap(loaded.target.model as? Qwen35)
        let loadedDrafter = try XCTUnwrap(
            loaded.drafter.model as? Qwen35VLMNextNDraftModel)
        XCTAssertEqual(loadedTarget.config.modelType, "qwen3_5")
        XCTAssertEqual(loadedDrafter.configuration.mtpNumHiddenLayers, 1)

        let targetNorm = try XCTUnwrap(
            loadedTarget.parameters().flattened().first {
                $0.0 == "language_model.model.norm.weight"
            }?.1)
        let drafterNorm = try XCTUnwrap(
            loadedDrafter.parameters().flattened().first {
                $0.0 == "mtp.norm.weight"
            }?.1)
        eval(targetNorm, drafterNorm)
        XCTAssertEqual(targetNorm.mean().item(Float.self), expectedNormValue, accuracy: 1e-6)
        XCTAssertEqual(drafterNorm.mean().item(Float.self), expectedNormValue, accuracy: 1e-6)
    }

    private func makeFactory(
        typeRegistry: ModelTypeRegistry<LanguageModel>? = nil
    ) -> VLMModelFactory {
        let typeRegistry = typeRegistry ?? ModelTypeRegistry<LanguageModel>(creators: [
            "qwen3_5": { data in
                Qwen35(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: data))
            }
        ])
        let processorRegistry = ProcessorTypeRegistry(creators: [
            "EmbeddedMTPTestProcessor": { _, _ in TestInputProcessor() }
        ])
        return VLMModelFactory(
            typeRegistry: typeRegistry,
            processorRegistry: processorRegistry,
            modelRegistry: VLMRegistry.shared,
            processorLoadingRegistry: VLMProcessorLoadingRegistry())
    }

    private func makeDirectory(configurationData: Data) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: "VLMEmbeddedMTPFactoryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try configurationData.write(to: directory.appending(component: "config.json"))
        try Data(#"{"processor_class":"EmbeddedMTPTestProcessor"}"#.utf8).write(
            to: directory.appending(component: "processor_config.json"))
        return directory
    }
}

private struct EmbeddedMTPTestTokenizerLoader: TokenizerLoader {
    func load(from _: URL) async throws -> any Tokenizer {
        TestTokenizer(vocabularySize: 16)
    }
}

private final class NonQwenVLMTarget: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(
        _ input: LMInput,
        cache _: [KVCache],
        state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }
}

private func qwenVLMConfigurationData(modelType: String = "qwen3_5") -> Data {
    Data(
        """
        {
          "model_type": "\(modelType)",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 16,
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 64,
            "tie_word_embeddings": true,
            "attention_bias": false,
            "full_attention_interval": 1,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "rope_parameters": {
              "type": "default",
              "rope_theta": 100000.0,
              "partial_rotary_factor": 0.25
            }
          },
          "vision_config": {
            "model_type": "qwen3_5_vit",
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "num_position_embeddings": 16
          }
        }
        """.utf8)
}
