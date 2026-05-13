"""Compatibility client orchestrating K8s data and the knowledge base."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rhoai_mcp.clients.base import CRDDefinition
from rhoai_mcp.composites.compatibility.knowledge_base import (
    check_runtime_gpu_compatibility,
    compare_versions,
    get_compatible_runtimes,
    get_gpu_architecture,
    get_rhoai_matrix,
)
from rhoai_mcp.composites.compatibility.models import (
    AffectedService,
    CompatibilityCheckResult,
    CompatibilityStatus,
    CrashDiagnosis,
    GPUNodeInfo,
    InstalledOperator,
    OperatorReadiness,
    RuntimeAlternative,
    UpgradeAssessment,
)

CSV_CRD = CRDDefinition(
    group="operators.coreos.com",
    version="v1alpha1",
    plural="clusterserviceversions",
    kind="ClusterServiceVersion",
)

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)

# CUDA error patterns and their diagnoses
_CUDA_ERROR_PATTERNS: list[tuple[str, str, bool]] = [
    (
        "no kernel image is available",
        "GPU compute capability mismatch -- the runtime binary was compiled "
        "for a higher compute capability than this GPU supports",
        True,
    ),
    (
        "CUDA driver version is insufficient",
        "NVIDIA driver is too old for the CUDA toolkit version required by this runtime",
        True,
    ),
    (
        "CUDA out of memory",
        "GPU out of memory -- this is a resource issue, not a compatibility issue",
        False,
    ),
    (
        "no CUDA-capable device",
        "No GPU device detected -- check that the GPU operator is installed "
        "and the node has working GPU hardware",
        True,
    ),
]


class CompatibilityClient:
    """Gathers cluster GPU/CUDA state and evaluates runtime compatibility."""

    def __init__(self, k8s: K8sClient, rhoai_version: str) -> None:
        self._k8s = k8s
        self._rhoai_version = rhoai_version

    def get_gpu_node_info(self) -> list[GPUNodeInfo]:
        """Gather GPU, CUDA, and driver info from all cluster nodes."""
        nodes = self._k8s.core_v1.list_node()
        result: list[GPUNodeInfo] = []

        for node in nodes.items:
            labels: dict[str, str] = node.metadata.labels or {}
            capacity: dict[str, str] = node.status.capacity or {}

            gpu_count = 0
            for key in ("nvidia.com/gpu", "amd.com/gpu"):
                if key in capacity:
                    gpu_count = int(capacity[key])
                    if gpu_count > 0:
                        break

            if gpu_count == 0:
                continue

            gpu_product = labels.get("nvidia.com/gpu.product") or labels.get(
                "nvidia.com/gpu-product"
            )

            cuda_major = labels.get("nvidia.com/cuda.runtime.major")
            cuda_minor = labels.get("nvidia.com/cuda.runtime.minor")
            cuda_version = f"{cuda_major}.{cuda_minor}" if cuda_major and cuda_minor else None

            compute_major = labels.get("nvidia.com/gpu.compute.major")
            compute_minor = labels.get("nvidia.com/gpu.compute.minor")
            compute_cap = (
                f"{compute_major}.{compute_minor}" if compute_major and compute_minor else None
            )

            driver_major = labels.get("nvidia.com/gpu.driver.major")
            driver_minor = labels.get("nvidia.com/gpu.driver.minor")
            driver_rev = labels.get("nvidia.com/gpu.driver.rev")
            driver_version = None
            if driver_major and driver_minor:
                driver_version = f"{driver_major}.{driver_minor}"
                if driver_rev:
                    driver_version += f".{driver_rev}"

            gpu_arch_name = None
            if gpu_product:
                arch = get_gpu_architecture(gpu_product)
                if arch:
                    gpu_arch_name = arch[0]
                    if not compute_cap:
                        compute_cap = arch[1]

            result.append(
                GPUNodeInfo(
                    node_name=node.metadata.name,
                    gpu_product=gpu_product,
                    gpu_count=gpu_count,
                    gpu_architecture=gpu_arch_name,
                    compute_capability=compute_cap,
                    cuda_version=cuda_version,
                    driver_version=driver_version,
                )
            )

        return result

    def check_runtime_compatibility(
        self,
        runtime_name: str | None = None,
        gpu_product: str | None = None,
        cuda_version: str | None = None,
    ) -> list[CompatibilityCheckResult]:
        """Check compatibility of runtimes against cluster hardware."""
        nodes = self.get_gpu_node_info()

        if not gpu_product and not cuda_version and nodes:
            unique_configs: dict[str, GPUNodeInfo] = {}
            for n in nodes:
                key = f"{n.gpu_product}|{n.cuda_version}|{n.driver_version}"
                if key not in unique_configs:
                    unique_configs[key] = n

            results: list[CompatibilityCheckResult] = []
            matrix = get_rhoai_matrix(self._rhoai_version)
            if not matrix:
                return [
                    CompatibilityCheckResult(
                        runtime_name=runtime_name or "unknown",
                        status=CompatibilityStatus.UNKNOWN,
                        issues=[
                            f"RHOAI version '{self._rhoai_version}' not in compatibility matrix"
                        ],
                    )
                ]

            runtimes_to_check = matrix.serving_runtimes
            if runtime_name:
                runtimes_to_check = tuple(
                    rt for rt in runtimes_to_check if rt.name == runtime_name
                )
                if not runtimes_to_check:
                    return [
                        CompatibilityCheckResult(
                            runtime_name=runtime_name,
                            status=CompatibilityStatus.UNKNOWN,
                            issues=[
                                f"Runtime '{runtime_name}' not found in "
                                f"RHOAI {self._rhoai_version} matrix"
                            ],
                        )
                    ]

            for rt in runtimes_to_check:
                for node_info in unique_configs.values():
                    result = check_runtime_gpu_compatibility(
                        runtime_name=rt.name,
                        gpu_product=node_info.gpu_product,
                        cuda_version=node_info.cuda_version,
                        driver_version=node_info.driver_version,
                        rhoai_version=self._rhoai_version,
                    )
                    results.append(result)

            return results

        matrix = get_rhoai_matrix(self._rhoai_version)
        if not matrix:
            return [
                CompatibilityCheckResult(
                    runtime_name=runtime_name or "unknown",
                    status=CompatibilityStatus.UNKNOWN,
                    issues=[
                        f"RHOAI version '{self._rhoai_version}' not in compatibility matrix"
                    ],
                )
            ]

        runtimes_to_check = matrix.serving_runtimes
        if runtime_name:
            runtimes_to_check = tuple(rt for rt in runtimes_to_check if rt.name == runtime_name)

        results = []
        for rt in runtimes_to_check:
            result = check_runtime_gpu_compatibility(
                runtime_name=rt.name,
                gpu_product=gpu_product,
                cuda_version=cuda_version,
                driver_version=None,
                rhoai_version=self._rhoai_version,
            )
            results.append(result)
        return results

    def diagnose_crash(self, service_name: str, namespace: str) -> CrashDiagnosis:
        """Diagnose whether a failing InferenceService is a compatibility issue."""
        from rhoai_mcp.domains.inference.client import InferenceClient

        inference = InferenceClient(self._k8s)
        evidence: list[str] = []
        is_compat = False
        root_cause = ""
        recommended_actions: list[str] = []

        try:
            isvc = inference.get_inference_service(service_name, namespace)
            for cond in isvc.conditions:
                if cond.status == "False" or cond.reason:
                    evidence.append(f"Condition {cond.type}: {cond.reason} - {cond.message}")
        except Exception as e:
            evidence.append(f"Could not fetch InferenceService: {e}")

        events = inference.get_inference_service_events(namespace, service_name)
        for ev in events:
            evidence.append(f"Event [{ev.get('reason')}]: {ev.get('message')}")

        logs = inference.get_inference_service_logs(namespace, service_name, tail_lines=50)
        if logs and not logs.startswith("No pods found") and not logs.startswith("Error"):
            for pattern, diagnosis, is_compatibility in _CUDA_ERROR_PATTERNS:
                if pattern.lower() in logs.lower():
                    root_cause = diagnosis
                    is_compat = is_compatibility
                    evidence.append(f"Log match: '{pattern}'")
                    break
            if not root_cause:
                evidence.append(f"Logs (tail): {logs[:500]}")

        pods = inference.get_inference_service_pods(namespace, service_name)
        compat_details = None
        node_name = None
        if pods:
            node_name = pods[0].get("node")

        if node_name and is_compat:
            nodes = self.get_gpu_node_info()
            target_node = next((n for n in nodes if n.node_name == node_name), None)
            if target_node:
                runtime = None
                try:
                    isvc = inference.get_inference_service(service_name, namespace)
                    runtime = isvc.runtime
                except Exception:
                    pass

                if runtime:
                    compat_details = check_runtime_gpu_compatibility(
                        runtime_name=runtime,
                        gpu_product=target_node.gpu_product,
                        cuda_version=target_node.cuda_version,
                        driver_version=target_node.driver_version,
                        rhoai_version=self._rhoai_version,
                    )
                    recommended_actions.extend(compat_details.recommendations)

                if target_node.gpu_product:
                    compatible_rts = get_compatible_runtimes(
                        self._rhoai_version,
                        gpu_product=target_node.gpu_product,
                        cuda_version=target_node.cuda_version,
                    )
                    if compatible_rts:
                        rt_names = [r.name for r in compatible_rts]
                        recommended_actions.append(
                            f"Compatible runtimes for this node: {', '.join(rt_names)}"
                        )

        if not root_cause:
            if any("CrashLoopBackOff" in e for e in evidence) or any(
                "BackOff" in e for e in evidence
            ):
                root_cause = (
                    "Container is crash-looping but no CUDA-specific error was found in logs. "
                    "This may be a configuration or resource issue rather than a "
                    "compatibility problem."
                )
            else:
                root_cause = "Unable to determine root cause from available evidence"

        if not recommended_actions:
            recommended_actions.append(
                "Check pod logs for more details: "
                f"oc logs -n {namespace} <pod-name> -c kserve-container"
            )

        return CrashDiagnosis(
            service_name=service_name,
            namespace=namespace,
            is_compatibility_issue=is_compat,
            root_cause=root_cause,
            evidence=evidence,
            compatibility_details=compat_details,
            recommended_actions=recommended_actions,
        )

    def assess_upgrade(self, target_version: str) -> UpgradeAssessment:
        """Assess compatibility impact of upgrading to a new RHOAI version."""
        from rhoai_mcp.domains.inference.client import InferenceClient

        current = self._rhoai_version
        current_matrix = get_rhoai_matrix(current)
        target_matrix = get_rhoai_matrix(target_version)

        if not target_matrix:
            return UpgradeAssessment(
                current_version=current,
                target_version=target_version,
                safe_to_upgrade=False,
                breaking_changes=[
                    f"RHOAI version '{target_version}' not found in compatibility matrix"
                ],
                recommendations=["Verify the target version is correct"],
            )

        nodes = self.get_gpu_node_info()
        inference = InferenceClient(self._k8s)

        affected: list[AffectedService] = []
        breaking: list[str] = []
        recommendations: list[str] = []

        namespaces = self._k8s.list_projects()
        for ns_resource in namespaces:
            ns_name = ns_resource.metadata.name
            services = inference.list_inference_services(ns_name)

            for svc in services:
                runtime_name = svc.get("runtime")
                if not runtime_name:
                    continue

                target_rt = None
                for rt in target_matrix.serving_runtimes:
                    if rt.name == runtime_name:
                        target_rt = rt
                        break

                if not target_rt:
                    affected.append(
                        AffectedService(
                            name=svc["name"],
                            namespace=ns_name,
                            runtime=runtime_name,
                            current_status=svc.get("status", "Unknown"),
                            impact=f"Runtime '{runtime_name}' not available in RHOAI {target_version}",
                            severity="high",
                        )
                    )
                    breaking.append(
                        f"Runtime '{runtime_name}' used by {svc['name']} "
                        f"is not available in RHOAI {target_version}"
                    )
                    continue

                if not target_rt.requires_gpu or target_rt.cuda_requirement is None:
                    continue

                for node in nodes:
                    if node.compute_capability and compare_versions(
                        node.compute_capability,
                        target_rt.cuda_requirement.min_compute_capability,
                    ) < 0:
                        affected.append(
                            AffectedService(
                                name=svc["name"],
                                namespace=ns_name,
                                runtime=runtime_name,
                                current_status=svc.get("status", "Unknown"),
                                impact=(
                                    f"Node {node.node_name} ({node.gpu_product}) has compute "
                                    f"capability {node.compute_capability}, but {runtime_name} "
                                    f"in RHOAI {target_version} requires "
                                    f"{target_rt.cuda_requirement.min_compute_capability}"
                                ),
                                severity="high",
                            )
                        )
                        breaking.append(
                            f"{runtime_name} in RHOAI {target_version} requires compute "
                            f"capability {target_rt.cuda_requirement.min_compute_capability}, "
                            f"but {node.gpu_product} on {node.node_name} only has "
                            f"{node.compute_capability}"
                        )

                    if node.cuda_version and compare_versions(
                        node.cuda_version, target_rt.cuda_requirement.cuda_min
                    ) < 0:
                        affected.append(
                            AffectedService(
                                name=svc["name"],
                                namespace=ns_name,
                                runtime=runtime_name,
                                current_status=svc.get("status", "Unknown"),
                                impact=(
                                    f"Node {node.node_name} has CUDA {node.cuda_version}, "
                                    f"but {runtime_name} in RHOAI {target_version} requires "
                                    f"CUDA {target_rt.cuda_requirement.cuda_min}+"
                                ),
                                severity="medium",
                            )
                        )

        # Check for 2.x -> 3.x upgrade path (not supported)
        current_major = current.split(".")[0]
        target_major = target_version.split(".")[0]
        if current_major != target_major:
            breaking.append(
                f"Direct upgrade from RHOAI {current_major}.x to "
                f"{target_major}.x is not supported. A fresh install is required."
            )

        if target_matrix.notes:
            for note in target_matrix.notes:
                breaking.append(note)

        if current_matrix and target_matrix:
            for target_rt in target_matrix.serving_runtimes:
                if not target_rt.requires_gpu or target_rt.cuda_requirement is None:
                    continue
                current_rt = None
                for crt in current_matrix.serving_runtimes:
                    if crt.name == target_rt.name:
                        current_rt = crt
                        break
                if (
                    current_rt
                    and current_rt.cuda_requirement is not None
                    and compare_versions(
                        target_rt.cuda_requirement.cuda_min,
                        current_rt.cuda_requirement.cuda_min,
                    ) > 0
                ):
                    recommendations.append(
                        f"{target_rt.name} CUDA requirement increased from "
                        f"{current_rt.cuda_requirement.cuda_min} to "
                        f"{target_rt.cuda_requirement.cuda_min}"
                    )

        safe = len([a for a in affected if a.severity == "high"]) == 0

        if not safe:
            recommendations.append(
                "Review affected services and consider using alternative runtimes "
                "or upgrading GPU drivers before upgrading RHOAI"
            )

        return UpgradeAssessment(
            current_version=current,
            target_version=target_version,
            affected_services=affected,
            safe_to_upgrade=safe,
            breaking_changes=breaking,
            recommendations=recommendations,
        )

    def find_compatible_runtimes(
        self,
        gpu_product: str | None = None,
        cuda_version: str | None = None,
        model_format: str | None = None,
    ) -> list[RuntimeAlternative]:
        """Find runtimes compatible with given hardware constraints."""
        if not gpu_product and not cuda_version:
            nodes = self.get_gpu_node_info()
            if nodes:
                gpu_product = nodes[0].gpu_product
                cuda_version = nodes[0].cuda_version

        compatible_specs = get_compatible_runtimes(
            self._rhoai_version,
            gpu_product=gpu_product,
            cuda_version=cuda_version,
        )

        if model_format:
            fmt_lower = model_format.lower()
            compatible_specs = [
                s for s in compatible_specs if fmt_lower in [f.lower() for f in s.supported_formats]
            ]

        results: list[RuntimeAlternative] = []
        for spec in compatible_specs:
            notes: list[str] = []
            if spec.known_issues:
                notes.extend(spec.known_issues)

            results.append(
                RuntimeAlternative(
                    runtime_name=spec.name,
                    display_name=spec.display_name,
                    compatibility_status=CompatibilityStatus.COMPATIBLE,
                    migration_notes=notes,
                    supported_formats=list(spec.supported_formats),
                )
            )

        return results

    def scan_installed_operators(self) -> list[InstalledOperator]:
        """Read ClusterServiceVersions from the cluster."""
        try:
            csvs = self._k8s.list_resources(CSV_CRD)
        except Exception:
            logger.debug("Failed to list ClusterServiceVersions")
            return []

        results: list[InstalledOperator] = []
        for csv in csvs:
            version = ""
            if hasattr(csv, "spec") and isinstance(csv.spec, dict):
                version = csv.spec.get("version", "")
            phase = ""
            if hasattr(csv, "status") and isinstance(csv.status, dict):
                phase = csv.status.get("phase", "")
            results.append(
                InstalledOperator(
                    name=csv.metadata.name,
                    namespace=csv.metadata.namespace or "",
                    version=version,
                    phase=phase,
                )
            )
        return results

    def check_operator_readiness(
        self, target_version: str | None = None,
    ) -> list[OperatorReadiness]:
        """Check installed operators against requirements for a RHOAI version."""
        version = target_version or self._rhoai_version
        matrix = get_rhoai_matrix(version)
        if not matrix:
            return []

        installed = self.scan_installed_operators()

        results: list[OperatorReadiness] = []
        for req in matrix.platform.operators:
            found = False
            found_version: str | None = None
            version_ok = True

            for inst in installed:
                if req.package_name and req.package_name.lower() in inst.name.lower():
                    found = True
                    found_version = inst.version
                    if (
                        req.min_version
                        and found_version
                        and compare_versions(found_version, req.min_version) < 0
                    ):
                        version_ok = False
                    break

            results.append(
                OperatorReadiness(
                    operator_name=req.name,
                    required=req.required,
                    feature=req.feature,
                    installed=found,
                    installed_version=found_version,
                    min_version=req.min_version,
                    version_ok=version_ok,
                    notes=req.notes,
                )
            )

        return results
