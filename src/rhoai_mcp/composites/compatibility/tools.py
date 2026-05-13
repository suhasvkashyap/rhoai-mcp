"""MCP tools for runtime version compatibility detection and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.compatibility.client import CompatibilityClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register runtime compatibility tools with the MCP server."""

    @mcp.tool()
    def check_runtime_compatibility(
        runtime_name: str | None = None,
        gpu_product: str | None = None,
        cuda_version: str | None = None,
        rhoai_version: str | None = None,
    ) -> dict[str, Any]:
        """Check if a serving runtime is compatible with cluster GPU hardware and CUDA version.

        Evaluates compatibility between serving runtimes, CUDA drivers, and GPU
        compute capabilities. Auto-detects GPU and CUDA information from cluster
        nodes when not explicitly provided.

        Use this when an administrator asks which runtime works with their
        hardware, e.g., "Our hardware only supports CUDA 12.2; which vLLM
        runtime should I use for RHOAI 2.16?"

        Args:
            runtime_name: Specific runtime to check (e.g., "vllm-runtime").
                If None, checks all known runtimes for the RHOAI version.
            gpu_product: GPU product name (e.g., "NVIDIA-A100-SXM4-80GB").
                Auto-detected from cluster nodes if None.
            cuda_version: CUDA toolkit version (e.g., "12.4").
                Auto-detected from node labels if None.
            rhoai_version: RHOAI version to check against (e.g., "2.16").
                Uses configured version if None.

        Returns:
            Compatibility results per runtime with status, issues, and
            recommendations.
        """
        version = rhoai_version or server.config.rhoai_version
        client = CompatibilityClient(server.k8s, version)
        results = client.check_runtime_compatibility(
            runtime_name=runtime_name,
            gpu_product=gpu_product,
            cuda_version=cuda_version,
        )

        compatible = [r for r in results if r.status.value == "compatible"]
        incompatible = [r for r in results if r.status.value == "incompatible"]

        return {
            "rhoai_version": version,
            "results": [r.model_dump() for r in results],
            "summary": {
                "total_checks": len(results),
                "compatible": len(compatible),
                "incompatible": len(incompatible),
            },
        }

    @mcp.tool()
    def diagnose_compatibility_issue(
        service_name: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Diagnose whether a failing InferenceService is caused by a runtime version mismatch.

        Examines pod logs, events, and conditions to determine if a
        CrashLoopBackOff or failure is caused by CUDA/GPU/runtime
        incompatibility versus other issues (OOM, image pull, etc.).

        Use this when a model deployment is failing and the administrator
        suspects a version mismatch, e.g., "My model is stuck in
        CrashLoopBackOff, is it a CUDA issue?"

        Args:
            service_name: Name of the InferenceService that is failing.
            namespace: Namespace (project) containing the service.

        Returns:
            Diagnosis with root cause, evidence from logs/events,
            compatibility details, and recommended actions to resolve.
        """
        client = CompatibilityClient(server.k8s, server.config.rhoai_version)
        diagnosis = client.diagnose_crash(service_name, namespace)

        result: dict[str, Any] = {
            "service_name": diagnosis.service_name,
            "namespace": diagnosis.namespace,
            "is_compatibility_issue": diagnosis.is_compatibility_issue,
            "root_cause": diagnosis.root_cause,
            "evidence": diagnosis.evidence,
            "recommended_actions": diagnosis.recommended_actions,
        }

        if diagnosis.compatibility_details:
            result["compatibility_details"] = diagnosis.compatibility_details.model_dump()

        return result

    @mcp.tool()
    def assess_upgrade_compatibility(
        target_version: str,
        current_version: str | None = None,
    ) -> dict[str, Any]:
        """Assess compatibility impact before upgrading RHOAI.

        Scans all deployed InferenceServices and checks whether their
        serving runtimes will remain compatible with the target RHOAI
        version's CUDA requirements on the current cluster hardware.

        Use this before a platform upgrade, e.g., "We're planning to
        upgrade to RHOAI 3.3. Will our current model deployments still
        work?"

        Args:
            target_version: RHOAI version to upgrade to (e.g., "3.3").
            current_version: Current RHOAI version. Uses configured
                version if None.

        Returns:
            Upgrade assessment with affected services, breaking changes,
            safety verdict, and migration recommendations.
        """
        from rhoai_mcp.composites.compatibility.knowledge_base import get_rhoai_matrix

        version = current_version or server.config.rhoai_version
        client = CompatibilityClient(server.k8s, version)
        assessment = client.assess_upgrade(target_version)
        operator_checks = client.check_operator_readiness(target_version)

        target_matrix = get_rhoai_matrix(target_version)
        platform_info = None
        if target_matrix:
            required_ops = []
            optional_ops = []
            for op in target_matrix.platform.operators:
                entry = {
                    "name": op.name,
                    "feature": op.feature,
                    "notes": op.notes,
                }
                if op.min_version:
                    entry["min_version"] = op.min_version
                if op.required:
                    required_ops.append(entry)
                else:
                    optional_ops.append(entry)
            platform_info = {
                "openshift_versions": list(target_matrix.platform.openshift_versions),
                "required_operators": required_ops,
                "optional_operators": optional_ops,
            }

        return {
            "current_version": assessment.current_version,
            "target_version": assessment.target_version,
            "safe_to_upgrade": assessment.safe_to_upgrade,
            "affected_services": [s.model_dump() for s in assessment.affected_services],
            "breaking_changes": assessment.breaking_changes,
            "recommendations": assessment.recommendations,
            "target_platform_requirements": platform_info,
            "operator_readiness": [c.model_dump() for c in operator_checks],
            "missing_operators": [
                {"name": c.operator_name, "feature": c.feature, "notes": c.notes}
                for c in operator_checks
                if c.required and not c.installed
            ]
            or None,
            "summary": {
                "total_affected": len(assessment.affected_services),
                "high_severity": len(
                    [s for s in assessment.affected_services if s.severity == "high"]
                ),
                "medium_severity": len(
                    [s for s in assessment.affected_services if s.severity == "medium"]
                ),
                "operators_ready": all(
                    c.installed and c.version_ok for c in operator_checks if c.required
                ),
            },
        }

    @mcp.tool()
    def find_compatible_runtimes(
        gpu_product: str | None = None,
        cuda_version: str | None = None,
        model_format: str | None = None,
    ) -> dict[str, Any]:
        """Find serving runtimes compatible with cluster hardware.

        Lists all serving runtimes that work with the specified (or
        auto-detected) GPU hardware and CUDA version. Optionally filters
        by model format. Use this when switching runtimes after a
        compatibility issue is detected.

        Args:
            gpu_product: GPU product name. Auto-detected from cluster
                if None.
            cuda_version: CUDA version. Auto-detected from node labels
                if None.
            model_format: Model format to filter by (e.g., "vLLM",
                "pytorch"). None returns all compatible runtimes.

        Returns:
            List of compatible runtimes with migration notes and format
            support.
        """
        client = CompatibilityClient(server.k8s, server.config.rhoai_version)
        alternatives = client.find_compatible_runtimes(
            gpu_product=gpu_product,
            cuda_version=cuda_version,
            model_format=model_format,
        )

        return {
            "rhoai_version": server.config.rhoai_version,
            "gpu_product": gpu_product,
            "cuda_version": cuda_version,
            "model_format": model_format,
            "compatible_runtimes": [a.model_dump() for a in alternatives],
            "total": len(alternatives),
        }

    @mcp.tool()
    def get_cluster_gpu_info() -> dict[str, Any]:
        """Get GPU hardware and CUDA driver information for all cluster nodes.

        Returns detailed GPU information including product names, compute
        capabilities, CUDA versions, and driver versions for each GPU
        node in the cluster. Use this to understand what hardware is
        available before running compatibility checks.

        Returns:
            Per-node GPU information with architecture details and CUDA
            versions.
        """
        client = CompatibilityClient(server.k8s, server.config.rhoai_version)
        nodes = client.get_gpu_node_info()

        products = list({n.gpu_product for n in nodes if n.gpu_product})
        cuda_versions = list({n.cuda_version for n in nodes if n.cuda_version})

        return {
            "gpu_nodes": [n.model_dump() for n in nodes],
            "total_gpu_nodes": len(nodes),
            "total_gpus": sum(n.gpu_count for n in nodes),
            "unique_gpu_products": products,
            "cuda_versions": cuda_versions,
            "heterogeneous": len(products) > 1,
        }

    @mcp.tool()
    def check_operator_readiness(
        target_version: str | None = None,
    ) -> dict[str, Any]:
        """Check which required operators are installed on the cluster.

        Scans installed ClusterServiceVersions and compares them against
        the operator requirements for a RHOAI version. Reports which
        required operators are missing, which optional ones could be
        added, and whether installed versions meet minimums.

        Use this before an install or upgrade, e.g., "What operators do
        I need for RHOAI 3.3?" or "Am I missing any prerequisites?"

        Args:
            target_version: RHOAI version to check against (e.g., "3.3").
                Uses configured version if None.

        Returns:
            Per-operator readiness with installed status, version check,
            and what each operator enables.
        """
        version = target_version or server.config.rhoai_version
        client = CompatibilityClient(server.k8s, version)
        checks = client.check_operator_readiness(version)

        missing_required = [c for c in checks if c.required and not c.installed]
        missing_optional = [c for c in checks if not c.required and not c.installed]
        version_issues = [c for c in checks if not c.version_ok]

        return {
            "rhoai_version": version,
            "operators": [c.model_dump() for c in checks],
            "summary": {
                "total_required": len([c for c in checks if c.required]),
                "total_optional": len([c for c in checks if not c.required]),
                "missing_required": len(missing_required),
                "missing_optional": len(missing_optional),
                "version_issues": len(version_issues),
                "ready": len(missing_required) == 0 and len(version_issues) == 0,
            },
            "missing_required": [
                {"name": c.operator_name, "feature": c.feature, "notes": c.notes}
                for c in missing_required
            ]
            if missing_required
            else None,
            "missing_optional": [
                {"name": c.operator_name, "feature": c.feature, "notes": c.notes}
                for c in missing_optional
            ]
            if missing_optional
            else None,
        }
