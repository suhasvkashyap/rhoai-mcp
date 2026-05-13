"""GPU/CUDA/runtime compatibility matrix and lookup functions.

This module contains the static compatibility data for RHOAI serving
runtimes, CUDA toolkit versions, GPU architectures, and driver versions.

Data sources (verified May 2026):
- RHOAI 2.x Supported Configs: https://access.redhat.com/articles/rhoai-supported-configs
- RHOAI 3.x Supported Configs: https://access.redhat.com/articles/rhoai-supported-configs-3.x
- NVIDIA GPU Operator platform support: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/platform-support.html
- RHOAI release notes per version
- Red Hat AI Inference Server supported configurations

To update: add a new RHOAIVersionMatrix entry when a new RHOAI version
ships. Verify CUDA version from the vLLM container image metadata
and driver requirements from the GPU Operator release notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rhoai_mcp.composites.compatibility.models import (
    CompatibilityCheckResult,
    CompatibilityStatus,
)

# ---------------------------------------------------------------------------
# GPU Architecture definitions (from NVIDIA CUDA documentation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUArchitecture:
    """NVIDIA GPU architecture definition."""

    name: str
    compute_capability: str
    representative_gpus: tuple[str, ...]


GPU_ARCHITECTURES: dict[str, GPUArchitecture] = {
    "pascal": GPUArchitecture(
        name="Pascal",
        compute_capability="6.0",
        representative_gpus=("P100", "P40"),
    ),
    "volta": GPUArchitecture(
        name="Volta",
        compute_capability="7.0",
        representative_gpus=("V100",),
    ),
    "turing": GPUArchitecture(
        name="Turing",
        compute_capability="7.5",
        representative_gpus=("T4", "RTX-6000"),
    ),
    "ampere": GPUArchitecture(
        name="Ampere",
        compute_capability="8.0",
        representative_gpus=("A100", "A30", "A10", "A10G", "A2", "A16", "A40"),
    ),
    "ada_lovelace": GPUArchitecture(
        name="Ada Lovelace",
        compute_capability="8.9",
        representative_gpus=("L4", "L20", "L40", "L40S"),
    ),
    "hopper": GPUArchitecture(
        name="Hopper",
        compute_capability="9.0",
        representative_gpus=("H100", "H200", "H20", "GH200"),
    ),
    "blackwell": GPUArchitecture(
        name="Blackwell",
        compute_capability="10.0",
        representative_gpus=("B100", "B200", "B300", "GB200", "GB300"),
    ),
}


# ---------------------------------------------------------------------------
# Quantization support by architecture (from Red Hat AI Inference Server docs)
# ---------------------------------------------------------------------------

QUANTIZATION_SUPPORT: dict[str, dict[str, bool]] = {
    "turing": {"fp16": True, "int8": True, "int4": False, "fp8_w8a8": False, "nvfp4": False},
    "ampere": {"fp16": True, "int8": True, "int4": True, "fp8_w8a8": False, "nvfp4": False},
    "ada_lovelace": {"fp16": True, "int8": True, "int4": True, "fp8_w8a8": True, "nvfp4": False},
    "hopper": {"fp16": True, "int8": True, "int4": True, "fp8_w8a8": True, "nvfp4": False},
    "blackwell": {"fp16": True, "int8": False, "int4": True, "fp8_w8a8": True, "nvfp4": True},
}


# ---------------------------------------------------------------------------
# Serving runtime and version matrix dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CUDARequirement:
    """CUDA toolkit and driver requirements for a serving runtime."""

    cuda_min: str
    cuda_max: str | None = None
    driver_min: str = "525.60.13"
    min_compute_capability: str = "7.5"


@dataclass(frozen=True)
class ServingRuntimeSpec:
    """Compatibility specification for a serving runtime in a given RHOAI version."""

    name: str
    display_name: str
    container_image: str = ""
    requires_gpu: bool = True
    cuda_requirement: CUDARequirement | None = None
    supported_formats: tuple[str, ...] = ()
    known_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorRequirement:
    """A single operator dependency for a RHOAI version."""

    name: str
    package_name: str = ""
    required: bool = True
    feature: str = ""
    min_version: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PlatformRequirement:
    """Platform-level requirements for a RHOAI version."""

    openshift_versions: tuple[str, ...] = ()
    operators: tuple[OperatorRequirement, ...] = ()


@dataclass(frozen=True)
class RHOAIVersionMatrix:
    """Compatibility matrix for a specific RHOAI version."""

    rhoai_version: str
    serving_runtimes: tuple[ServingRuntimeSpec, ...]
    default_cuda_toolkit: str
    platform: PlatformRequirement = field(default_factory=PlatformRequirement)
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# RHOAI version compatibility matrices
#
# Versions covered: 2.16 (EUS), 2.25 (last 2.x), 3.3 (latest 3.x)
# Source: Red Hat OpenShift AI Supported Configurations and release notes
# ---------------------------------------------------------------------------

COMPATIBILITY_MATRIX: dict[str, RHOAIVersionMatrix] = {
    "2.16": RHOAIVersionMatrix(
        rhoai_version="2.16",
        default_cuda_toolkit="12.1",
        platform=PlatformRequirement(
            openshift_versions=("4.14", "4.16", "4.17", "4.18", "4.19"),
            operators=(
                OperatorRequirement(
                    name="Red Hat OpenShift AI Operator",
                    package_name="rhods-operator",
                    required=True,
                    feature="core platform",
                ),
                OperatorRequirement(
                    name="Red Hat OpenShift Serverless",
                    package_name="serverless-operator",
                    required=True,
                    feature="model serving (KServe)",
                    notes="Required for KServe-based model serving in 2.x",
                ),
                OperatorRequirement(
                    name="Red Hat OpenShift Service Mesh v2",
                    package_name="servicemeshoperator",
                    required=True,
                    feature="model serving (KServe)",
                    notes="Required for KServe networking in 2.x; v3 is NOT supported",
                ),
                OperatorRequirement(
                    name="Node Feature Discovery Operator",
                    package_name="nfd",
                    required=False,
                    feature="GPU node labeling",
                    notes="Labels nodes with GPU product, CUDA version, and driver info",
                ),
                OperatorRequirement(
                    name="NVIDIA GPU Operator",
                    package_name="gpu-operator-certified",
                    required=False,
                    feature="NVIDIA GPU support",
                    notes="Installs drivers, device plugin, and container toolkit for NVIDIA GPUs",
                ),
            ),
        ),
        serving_runtimes=(
            ServingRuntimeSpec(
                name="vllm-runtime",
                display_name="vLLM ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-vllm-cuda-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.1",
                    driver_min="525.60.13",
                    min_compute_capability="7.5",
                ),
                supported_formats=("vLLM",),
            ),
            ServingRuntimeSpec(
                name="tgis-runtime",
                display_name="TGIS Standalone ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-tgis-serving-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.1",
                    driver_min="525.60.13",
                    min_compute_capability="7.5",
                ),
                supported_formats=("pytorch",),
            ),
            ServingRuntimeSpec(
                name="caikit-tgis-runtime",
                display_name="Caikit-TGIS ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-caikit-tgis-serving-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.1",
                    driver_min="525.60.13",
                    min_compute_capability="7.5",
                ),
                supported_formats=("caikit",),
            ),
            ServingRuntimeSpec(
                name="openvino-runtime",
                display_name="OpenVINO Model Server",
                requires_gpu=False,
                cuda_requirement=None,
                supported_formats=("onnx", "openvino"),
                known_issues=(
                    "OpenVINO is CPU-based (Intel); does not use NVIDIA GPUs or CUDA",
                ),
            ),
        ),
        notes=(
            "RHOAI 2.16 is an Extended Update Support (EUS) release",
            "TGIS standalone runtime is available but deprecated; removed in 3.0",
            "Training images use CUDA 12.1 with PyTorch 2.4.1",
        ),
    ),
    "2.25": RHOAIVersionMatrix(
        rhoai_version="2.25",
        default_cuda_toolkit="12.6",
        platform=PlatformRequirement(
            openshift_versions=("4.16", "4.17", "4.18", "4.19", "4.20"),
            operators=(
                OperatorRequirement(
                    name="Red Hat OpenShift AI Operator",
                    package_name="rhods-operator",
                    required=True,
                    feature="core platform",
                ),
                OperatorRequirement(
                    name="Node Feature Discovery Operator",
                    package_name="nfd",
                    required=False,
                    feature="GPU node labeling",
                    notes="Labels nodes with GPU product, CUDA version, and driver info",
                ),
                OperatorRequirement(
                    name="NVIDIA GPU Operator",
                    package_name="gpu-operator-certified",
                    required=False,
                    feature="NVIDIA GPU support",
                    notes="Installs drivers, device plugin, and container toolkit",
                ),
            ),
        ),
        serving_runtimes=(
            ServingRuntimeSpec(
                name="vllm-runtime",
                display_name="vLLM ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-vllm-cuda-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.4",
                    driver_min="535.129.03",
                    min_compute_capability="7.5",
                ),
                supported_formats=("vLLM",),
                known_issues=(
                    "T4 (Turing) GPUs do not support INT4 quantization; use FP16 instead",
                ),
            ),
            ServingRuntimeSpec(
                name="caikit-tgis-runtime",
                display_name="Caikit-TGIS ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-caikit-tgis-serving-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.4",
                    driver_min="535.129.03",
                    min_compute_capability="7.5",
                ),
                supported_formats=("caikit",),
            ),
            ServingRuntimeSpec(
                name="openvino-runtime",
                display_name="OpenVINO Model Server",
                requires_gpu=False,
                cuda_requirement=None,
                supported_formats=("onnx", "openvino"),
                known_issues=(
                    "OpenVINO is CPU-based (Intel); does not use NVIDIA GPUs or CUDA",
                ),
            ),
        ),
        notes=(
            "Last 2.x release; no direct upgrade path to 3.x",
            "TGIS standalone runtime removed; use Caikit-TGIS or vLLM",
            "Workbench images use CUDA 12.6 with Python 3.11",
        ),
    ),
    "3.3": RHOAIVersionMatrix(
        rhoai_version="3.3",
        default_cuda_toolkit="12.8",
        platform=PlatformRequirement(
            openshift_versions=("4.19", "4.20", "4.21"),
            operators=(
                OperatorRequirement(
                    name="Red Hat OpenShift AI Operator",
                    package_name="rhods-operator",
                    required=True,
                    feature="core platform",
                ),
                OperatorRequirement(
                    name="Cert Manager Operator",
                    package_name="openshift-cert-manager-operator",
                    required=True,
                    feature="KServe / service mesh certificates",
                    notes="New requirement in 3.x; not required in 2.x",
                ),
                OperatorRequirement(
                    name="Jobset Operator",
                    package_name="jobset-operator",
                    required=True,
                    feature="training jobs",
                    notes="New requirement in 3.3; not required in 3.0 or 3.2",
                ),
                OperatorRequirement(
                    name="Node Feature Discovery Operator",
                    package_name="nfd",
                    required=False,
                    feature="GPU node labeling",
                    notes="Labels nodes with GPU product, CUDA version, and driver info; "
                    "required before installing any GPU operator",
                ),
                OperatorRequirement(
                    name="NVIDIA GPU Operator",
                    package_name="gpu-operator-certified",
                    required=False,
                    feature="NVIDIA GPU support",
                    min_version="24.3",
                    notes="Installs drivers, device plugin, container toolkit; "
                    "requires NFD Operator installed first",
                ),
                OperatorRequirement(
                    name="AMD GPU Operator",
                    package_name="amd-gpu-operator",
                    required=False,
                    feature="AMD GPU support (ROCm)",
                    notes="For AMD Instinct MI300X; requires NFD and KMM operators",
                ),
                OperatorRequirement(
                    name="Intel Gaudi Base Operator",
                    package_name="habana-ai-operator",
                    required=False,
                    feature="Intel Gaudi HPU support",
                    notes="For Intel Gaudi accelerators; does not use CUDA",
                ),
                OperatorRequirement(
                    name="Leader Worker Set",
                    package_name="leader-worker-set",
                    required=False,
                    feature="distributed inference (llm-d)",
                    notes="Required only for llm-d distributed inference; "
                    "needs OpenShift 4.20+",
                ),
                OperatorRequirement(
                    name="Red Hat Authorino Operator",
                    package_name="authorino-operator",
                    required=False,
                    feature="model serving authorization",
                    notes="Only the Red Hat Connectivity Link component is supported",
                ),
                OperatorRequirement(
                    name="KubeRay Operator",
                    package_name="kuberay-operator",
                    required=False,
                    feature="distributed training (Ray)",
                    notes="Replaces CodeFlare Operator which was removed in 3.0",
                ),
            ),
        ),
        serving_runtimes=(
            ServingRuntimeSpec(
                name="vllm-runtime",
                display_name="vLLM ServingRuntime for KServe (CUDA)",
                container_image="registry.redhat.io/rhaiis/vllm-cuda-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.8",
                    driver_min="535.129.03",
                    min_compute_capability="7.5",
                ),
                supported_formats=("vLLM",),
                known_issues=(
                    "T4 (Turing): no optimized INT4 kernel; no FP8 W8A8 support",
                    "Ampere (A100/A10): no FP8 W8A8 support (FP8 W8A16 via Marlin OK)",
                    "Blackwell (B200/B300): no INT8 support in vLLM; use FP8 or NVFP4",
                    "On nodes with driver 580.x (CUDA 13.0), may see "
                    "cudaErrorSystemDriverMismatch; set LD_LIBRARY_PATH=/usr/lib64 "
                    "in entrypoint as workaround",
                ),
            ),
            ServingRuntimeSpec(
                name="vllm-cpu-runtime",
                display_name="vLLM ServingRuntime for KServe (CPU)",
                container_image="registry.redhat.io/rhoai/odh-vllm-cpu-rhel9",
                requires_gpu=False,
                cuda_requirement=None,
                supported_formats=("vLLM",),
                known_issues=(
                    "CPU-only; significantly slower than GPU-accelerated serving",
                ),
            ),
            ServingRuntimeSpec(
                name="vllm-gaudi-runtime",
                display_name="vLLM ServingRuntime for KServe (Intel Gaudi)",
                container_image="registry.redhat.io/rhoai/odh-vllm-gaudi-rhel9",
                requires_gpu=False,
                cuda_requirement=None,
                supported_formats=("vLLM",),
                known_issues=(
                    "Requires Intel Gaudi HPU; does not use NVIDIA GPUs or CUDA",
                ),
            ),
            ServingRuntimeSpec(
                name="caikit-tgis-runtime",
                display_name="Caikit-TGIS ServingRuntime for KServe",
                container_image="registry.redhat.io/rhoai/odh-caikit-tgis-serving-rhel9",
                requires_gpu=True,
                cuda_requirement=CUDARequirement(
                    cuda_min="12.8",
                    driver_min="535.129.03",
                    min_compute_capability="7.5",
                ),
                supported_formats=("caikit",),
            ),
            ServingRuntimeSpec(
                name="openvino-runtime",
                display_name="OpenVINO Model Server",
                requires_gpu=False,
                cuda_requirement=None,
                supported_formats=("onnx", "openvino"),
                known_issues=(
                    "OpenVINO is CPU-based (Intel); does not use NVIDIA GPUs or CUDA",
                ),
            ),
        ),
        notes=(
            "First 3.x version requiring Cert Manager and Jobset operators",
            "TGIS standalone removed; CodeFlare removed (use KubeRay directly)",
            "Training images use CUDA 12.8 with PyTorch 2.8.0 and Python 3.12",
            "vLLM v0.13.0 bundled; built on CUDA 12.8",
            "No direct upgrade path from 2.x to 3.x; fresh install required",
            "llm-d distributed inference requires OpenShift 4.20+",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def compare_versions(a: str, b: str) -> int:
    """Compare two version strings numerically.

    Returns -1 if a < b, 0 if a == b, 1 if a > b.
    """
    a_parts = [int(x) for x in a.split(".")]
    b_parts = [int(x) for x in b.split(".")]
    for av, bv in zip(a_parts, b_parts, strict=False):
        if av < bv:
            return -1
        if av > bv:
            return 1
    if len(a_parts) < len(b_parts):
        return -1
    if len(a_parts) > len(b_parts):
        return 1
    return 0


def get_gpu_architecture(gpu_product: str) -> tuple[str, str] | None:
    """Match a GPU product name to its architecture.

    Args:
        gpu_product: GPU product string from node label
            (e.g., "NVIDIA-A100-SXM4-80GB", "Tesla-T4").

    Returns:
        Tuple of (architecture_name, compute_capability) or None.
    """
    product_upper = gpu_product.upper()
    for arch in GPU_ARCHITECTURES.values():
        for gpu in arch.representative_gpus:
            if gpu.upper() in product_upper:
                return arch.name, arch.compute_capability
    return None


def get_rhoai_matrix(rhoai_version: str) -> RHOAIVersionMatrix | None:
    """Get the compatibility matrix for a specific RHOAI version."""
    return COMPATIBILITY_MATRIX.get(rhoai_version)


def get_compatible_runtimes(
    rhoai_version: str,
    gpu_product: str | None = None,
    cuda_version: str | None = None,
) -> list[ServingRuntimeSpec]:
    """Return GPU-capable runtimes compatible with the given hardware.

    Args:
        rhoai_version: RHOAI version string (e.g., "2.16").
        gpu_product: GPU product name. If None, skips GPU filtering.
        cuda_version: CUDA version. If None, skips CUDA filtering.

    Returns:
        List of compatible runtime specs (GPU-based runtimes only).
    """
    matrix = COMPATIBILITY_MATRIX.get(rhoai_version)
    if not matrix:
        return []

    compute_cap = None
    if gpu_product:
        arch = get_gpu_architecture(gpu_product)
        if arch:
            compute_cap = arch[1]

    results = []
    for rt in matrix.serving_runtimes:
        if not rt.requires_gpu or rt.cuda_requirement is None:
            continue

        req = rt.cuda_requirement

        if compute_cap and compare_versions(compute_cap, req.min_compute_capability) < 0:
            continue

        if cuda_version:
            if compare_versions(cuda_version, req.cuda_min) < 0:
                continue
            if req.cuda_max and compare_versions(cuda_version, req.cuda_max) > 0:
                continue

        results.append(rt)

    return results


def get_all_runtimes(rhoai_version: str) -> list[ServingRuntimeSpec]:
    """Return all runtimes (GPU and CPU) for a RHOAI version."""
    matrix = COMPATIBILITY_MATRIX.get(rhoai_version)
    if not matrix:
        return []
    return list(matrix.serving_runtimes)


def check_runtime_gpu_compatibility(
    runtime_name: str,
    gpu_product: str | None,
    cuda_version: str | None,
    driver_version: str | None,
    rhoai_version: str,
) -> CompatibilityCheckResult:
    """Check if a specific runtime is compatible with specific hardware.

    Args:
        runtime_name: Serving runtime name (e.g., "vllm-runtime").
        gpu_product: GPU product from node label.
        cuda_version: CUDA toolkit version.
        driver_version: NVIDIA driver version.
        rhoai_version: RHOAI version.

    Returns:
        CompatibilityCheckResult with status, issues, and recommendations.
    """
    matrix = COMPATIBILITY_MATRIX.get(rhoai_version)
    if not matrix:
        return CompatibilityCheckResult(
            runtime_name=runtime_name,
            gpu_product=gpu_product,
            cuda_version=cuda_version,
            driver_version=driver_version,
            status=CompatibilityStatus.UNKNOWN,
            issues=[f"RHOAI version '{rhoai_version}' not found in compatibility matrix"],
            recommendations=["Check that the RHOAI version is correct"],
        )

    rt_spec = None
    for rt in matrix.serving_runtimes:
        if rt.name == runtime_name:
            rt_spec = rt
            break

    if not rt_spec:
        return CompatibilityCheckResult(
            runtime_name=runtime_name,
            gpu_product=gpu_product,
            cuda_version=cuda_version,
            driver_version=driver_version,
            status=CompatibilityStatus.UNKNOWN,
            issues=[
                f"Runtime '{runtime_name}' not found in RHOAI {rhoai_version} compatibility matrix"
            ],
            recommendations=["Verify the runtime name is correct"],
        )

    if not rt_spec.requires_gpu or rt_spec.cuda_requirement is None:
        return CompatibilityCheckResult(
            runtime_name=runtime_name,
            runtime_display_name=rt_spec.display_name,
            gpu_product=gpu_product,
            status=CompatibilityStatus.COMPATIBLE,
            issues=[],
            recommendations=[
                f"{rt_spec.display_name} is CPU-based and does not require GPU/CUDA"
            ],
        )

    issues: list[str] = []
    recommendations: list[str] = []
    req = rt_spec.cuda_requirement

    gpu_arch_name = None
    compute_cap = None
    if gpu_product:
        arch = get_gpu_architecture(gpu_product)
        if arch:
            gpu_arch_name, compute_cap = arch

    if compute_cap and compare_versions(compute_cap, req.min_compute_capability) < 0:
        issues.append(
            f"GPU compute capability {compute_cap} is below the minimum "
            f"{req.min_compute_capability} required by {rt_spec.display_name}"
        )
        compatible_rts = get_compatible_runtimes(rhoai_version, gpu_product=gpu_product)
        if compatible_rts:
            alt_names = [r.name for r in compatible_rts]
            recommendations.append(f"Use an alternative runtime: {', '.join(alt_names)}")
        else:
            recommendations.append(
                "No compatible GPU runtimes for this hardware. "
                "Consider a CPU-based runtime (openvino-runtime, vllm-cpu-runtime) "
                "or upgrading GPU hardware."
            )

    if cuda_version:
        if compare_versions(cuda_version, req.cuda_min) < 0:
            issues.append(
                f"CUDA version {cuda_version} is below the minimum "
                f"{req.cuda_min} required by {rt_spec.display_name}"
            )
            recommendations.append(
                f"Update the NVIDIA GPU Operator to get CUDA {req.cuda_min}+ drivers, "
                f"or use a runtime with lower CUDA requirements"
            )
        if req.cuda_max and compare_versions(cuda_version, req.cuda_max) > 0:
            issues.append(
                f"CUDA version {cuda_version} exceeds the maximum "
                f"{req.cuda_max} supported by {rt_spec.display_name}"
            )
            recommendations.append(
                f"Downgrade CUDA toolkit to {req.cuda_max} or earlier"
            )

    if driver_version and compare_versions(driver_version, req.driver_min) < 0:
        issues.append(
            f"NVIDIA driver version {driver_version} is below the minimum "
            f"{req.driver_min} required by {rt_spec.display_name}"
        )
        recommendations.append(
            f"Update the NVIDIA GPU Operator to get driver {req.driver_min} or later"
        )

    for known_issue in rt_spec.known_issues:
        if gpu_product and gpu_arch_name:
            arch_lower = gpu_arch_name.lower()
            if arch_lower in known_issue.lower() or gpu_product.upper() in known_issue.upper():
                issues.append(f"Known issue: {known_issue}")

    if issues:
        has_blocking = any(
            "below the minimum" in i or "compute capability" in i
            for i in issues
            if not i.startswith("Known issue")
        )
        status = CompatibilityStatus.INCOMPATIBLE if has_blocking else CompatibilityStatus.DEGRADED
    elif not gpu_product and not cuda_version:
        status = CompatibilityStatus.UNKNOWN
    else:
        status = CompatibilityStatus.COMPATIBLE

    return CompatibilityCheckResult(
        runtime_name=runtime_name,
        runtime_display_name=rt_spec.display_name,
        gpu_product=gpu_product,
        gpu_architecture=gpu_arch_name,
        compute_capability=compute_cap,
        cuda_version=cuda_version,
        driver_version=driver_version,
        status=status,
        issues=issues,
        recommendations=recommendations,
    )
