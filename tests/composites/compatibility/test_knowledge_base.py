"""Tests for the runtime compatibility knowledge base."""

import pytest

from rhoai_mcp.composites.compatibility.knowledge_base import (
    COMPATIBILITY_MATRIX,
    GPU_ARCHITECTURES,
    check_runtime_gpu_compatibility,
    compare_versions,
    get_all_runtimes,
    get_compatible_runtimes,
    get_gpu_architecture,
    get_rhoai_matrix,
)
from rhoai_mcp.composites.compatibility.models import CompatibilityStatus


class TestCompareVersions:
    def test_equal(self) -> None:
        assert compare_versions("12.4", "12.4") == 0

    def test_less_than(self) -> None:
        assert compare_versions("12.2", "12.4") == -1

    def test_greater_than(self) -> None:
        assert compare_versions("13.0", "12.4") == 1

    def test_different_lengths(self) -> None:
        assert compare_versions("12", "12.0") == -1
        assert compare_versions("12.0", "12") == 1

    def test_major_difference(self) -> None:
        assert compare_versions("11.8", "12.1") == -1

    def test_three_part_version(self) -> None:
        assert compare_versions("525.60.13", "535.129.03") == -1
        assert compare_versions("535.129.03", "525.60.13") == 1


class TestGetGPUArchitecture:
    def test_a100(self) -> None:
        result = get_gpu_architecture("NVIDIA-A100-SXM4-80GB")
        assert result is not None
        assert result[0] == "Ampere"
        assert result[1] == "8.0"

    def test_t4(self) -> None:
        result = get_gpu_architecture("Tesla-T4")
        assert result is not None
        assert result[0] == "Turing"
        assert result[1] == "7.5"

    def test_v100(self) -> None:
        result = get_gpu_architecture("Tesla-V100-SXM2-32GB")
        assert result is not None
        assert result[0] == "Volta"
        assert result[1] == "7.0"

    def test_h100(self) -> None:
        result = get_gpu_architecture("NVIDIA-H100-SXM5-80GB")
        assert result is not None
        assert result[0] == "Hopper"
        assert result[1] == "9.0"

    def test_l40s(self) -> None:
        result = get_gpu_architecture("NVIDIA-L40S")
        assert result is not None
        assert result[0] == "Ada Lovelace"
        assert result[1] == "8.9"

    def test_b200(self) -> None:
        result = get_gpu_architecture("NVIDIA-B200")
        assert result is not None
        assert result[0] == "Blackwell"
        assert result[1] == "10.0"

    def test_unknown_gpu(self) -> None:
        result = get_gpu_architecture("Unknown-GPU-Model")
        assert result is None

    def test_case_insensitive(self) -> None:
        result = get_gpu_architecture("nvidia-a100-pcie-40gb")
        assert result is not None
        assert result[0] == "Ampere"


class TestGetRHOAIMatrix:
    def test_known_version_216(self) -> None:
        matrix = get_rhoai_matrix("2.16")
        assert matrix is not None
        assert matrix.rhoai_version == "2.16"
        assert len(matrix.serving_runtimes) > 0

    def test_known_version_33(self) -> None:
        matrix = get_rhoai_matrix("3.3")
        assert matrix is not None
        assert matrix.default_cuda_toolkit == "12.8"

    def test_unknown_version(self) -> None:
        assert get_rhoai_matrix("99.99") is None

    def test_all_versions_have_runtimes(self) -> None:
        for version, matrix in COMPATIBILITY_MATRIX.items():
            assert len(matrix.serving_runtimes) > 0, f"Version {version} has no runtimes"

    def test_33_has_platform_requirements(self) -> None:
        matrix = get_rhoai_matrix("3.3")
        assert matrix is not None
        assert "4.19" in matrix.platform.openshift_versions
        op_names = [op.name for op in matrix.platform.operators]
        assert "Cert Manager Operator" in op_names
        assert "Jobset Operator" in op_names

    def test_33_operators_have_features(self) -> None:
        matrix = get_rhoai_matrix("3.3")
        assert matrix is not None
        for op in matrix.platform.operators:
            assert op.feature, f"Operator {op.name} has no feature description"

    def test_33_has_required_and_optional_operators(self) -> None:
        matrix = get_rhoai_matrix("3.3")
        assert matrix is not None
        required = [op for op in matrix.platform.operators if op.required]
        optional = [op for op in matrix.platform.operators if not op.required]
        assert len(required) >= 3
        assert len(optional) >= 4

    def test_operator_package_names_set(self) -> None:
        matrix = get_rhoai_matrix("3.3")
        assert matrix is not None
        for op in matrix.platform.operators:
            assert op.package_name, f"Operator {op.name} has no package_name"

    def test_versions_covered(self) -> None:
        assert "2.16" in COMPATIBILITY_MATRIX
        assert "2.25" in COMPATIBILITY_MATRIX
        assert "3.3" in COMPATIBILITY_MATRIX


class TestGetCompatibleRuntimes:
    def test_all_gpu_runtimes_for_version(self) -> None:
        runtimes = get_compatible_runtimes("2.16")
        assert len(runtimes) > 0
        for rt in runtimes:
            assert rt.requires_gpu is True

    def test_cpu_runtimes_excluded(self) -> None:
        runtimes = get_compatible_runtimes("3.3")
        names = [r.name for r in runtimes]
        assert "openvino-runtime" not in names
        assert "vllm-cpu-runtime" not in names

    def test_filter_by_cuda_version(self) -> None:
        runtimes = get_compatible_runtimes("3.3", cuda_version="12.6")
        names = [r.name for r in runtimes]
        assert "vllm-runtime" not in names

    def test_a100_compatible_with_33(self) -> None:
        runtimes = get_compatible_runtimes(
            "3.3", gpu_product="NVIDIA-A100-SXM4-80GB", cuda_version="12.8"
        )
        names = [r.name for r in runtimes]
        assert "vllm-runtime" in names

    def test_unknown_version_returns_empty(self) -> None:
        runtimes = get_compatible_runtimes("99.99")
        assert runtimes == []


class TestGetAllRuntimes:
    def test_includes_cpu_runtimes(self) -> None:
        runtimes = get_all_runtimes("3.3")
        names = [r.name for r in runtimes]
        assert "openvino-runtime" in names
        assert "vllm-cpu-runtime" in names
        assert "vllm-runtime" in names

    def test_216_has_tgis(self) -> None:
        runtimes = get_all_runtimes("2.16")
        names = [r.name for r in runtimes]
        assert "tgis-runtime" in names

    def test_225_no_tgis(self) -> None:
        runtimes = get_all_runtimes("2.25")
        names = [r.name for r in runtimes]
        assert "tgis-runtime" not in names


class TestCheckRuntimeGPUCompatibility:
    def test_compatible_a100_vllm_216(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.4",
            driver_version="535.129.03",
            rhoai_version="2.16",
        )
        assert result.status == CompatibilityStatus.COMPATIBLE
        assert len(result.issues) == 0

    def test_incompatible_cuda_too_old_for_33(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.6",
            driver_version="535.129.03",
            rhoai_version="3.3",
        )
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert any("CUDA version" in i for i in result.issues)

    def test_incompatible_driver_too_old(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.8",
            driver_version="525.60.13",
            rhoai_version="3.3",
        )
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert any("driver version" in i for i in result.issues)

    def test_openvino_cpu_always_compatible(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="openvino-runtime",
            gpu_product="Tesla-T4",
            cuda_version="12.2",
            driver_version=None,
            rhoai_version="3.3",
        )
        assert result.status == CompatibilityStatus.COMPATIBLE
        assert any("CPU-based" in r for r in result.recommendations)

    def test_unknown_rhoai_version(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.4",
            driver_version=None,
            rhoai_version="99.99",
        )
        assert result.status == CompatibilityStatus.UNKNOWN

    def test_unknown_runtime(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="nonexistent-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.4",
            driver_version=None,
            rhoai_version="2.16",
        )
        assert result.status == CompatibilityStatus.UNKNOWN

    def test_no_gpu_or_cuda_returns_unknown(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product=None,
            cuda_version=None,
            driver_version=None,
            rhoai_version="2.16",
        )
        assert result.status == CompatibilityStatus.UNKNOWN

    def test_recommendations_present_on_incompatible(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.6",
            driver_version=None,
            rhoai_version="3.3",
        )
        assert len(result.recommendations) > 0

    def test_t4_known_issues_degraded(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="Tesla-T4",
            cuda_version="12.8",
            driver_version="535.129.03",
            rhoai_version="3.3",
        )
        assert result.status == CompatibilityStatus.DEGRADED
        assert any("Known issue" in i for i in result.issues)
        assert result.gpu_architecture == "Turing"

    def test_volta_below_min_compute_33(self) -> None:
        result = check_runtime_gpu_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="Tesla-V100-SXM2-32GB",
            cuda_version="12.8",
            driver_version="535.129.03",
            rhoai_version="3.3",
        )
        assert result.status == CompatibilityStatus.INCOMPATIBLE
        assert any("compute capability" in i for i in result.issues)


class TestGPUArchitectures:
    def test_all_architectures_have_gpus(self) -> None:
        for name, arch in GPU_ARCHITECTURES.items():
            assert len(arch.representative_gpus) > 0, f"{name} has no GPUs"

    def test_compute_capabilities_are_ordered(self) -> None:
        caps = [float(a.compute_capability) for a in GPU_ARCHITECTURES.values()]
        assert caps == sorted(caps)

    def test_supported_gpus_present(self) -> None:
        ampere = GPU_ARCHITECTURES["ampere"]
        assert "A100" in ampere.representative_gpus
        assert "A10" in ampere.representative_gpus
        assert "A2" in ampere.representative_gpus
