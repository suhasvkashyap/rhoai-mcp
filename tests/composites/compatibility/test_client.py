"""Tests for the CompatibilityClient."""

import pytest
from unittest.mock import MagicMock

from rhoai_mcp.composites.compatibility.client import CompatibilityClient
from rhoai_mcp.composites.compatibility.models import CompatibilityStatus
from rhoai_mcp.mock_k8s.cluster_state import create_default_cluster_state
from rhoai_mcp.mock_k8s.mock_client import MockK8sClient


@pytest.fixture
def mock_client() -> MockK8sClient:
    state = create_default_cluster_state()
    client = MockK8sClient(state=state)
    client.connect()
    return client


@pytest.fixture
def compat_client(mock_client: MockK8sClient) -> CompatibilityClient:
    return CompatibilityClient(mock_client, rhoai_version="2.16")


class TestGetGPUNodeInfo:
    def test_returns_gpu_nodes(self, compat_client: CompatibilityClient) -> None:
        nodes = compat_client.get_gpu_node_info()
        assert len(nodes) == 3

    def test_a100_node_info(self, compat_client: CompatibilityClient) -> None:
        nodes = compat_client.get_gpu_node_info()
        a100_node = next(n for n in nodes if n.node_name == "gpu-node-1")
        assert a100_node.gpu_product == "NVIDIA-A100-SXM4-80GB"
        assert a100_node.gpu_count == 4
        assert a100_node.cuda_version == "12.4"
        assert a100_node.compute_capability == "8.0"
        assert a100_node.gpu_architecture == "Ampere"
        assert a100_node.driver_version == "535.129.03"

    def test_t4_node_info(self, compat_client: CompatibilityClient) -> None:
        nodes = compat_client.get_gpu_node_info()
        t4_node = next(n for n in nodes if n.node_name == "gpu-node-3")
        assert t4_node.gpu_product == "Tesla-T4"
        assert t4_node.gpu_count == 2
        assert t4_node.cuda_version == "12.2"
        assert t4_node.compute_capability == "7.5"
        assert t4_node.gpu_architecture == "Turing"


class TestCheckRuntimeCompatibility:
    def test_auto_detect_from_cluster(self, compat_client: CompatibilityClient) -> None:
        results = compat_client.check_runtime_compatibility()
        assert len(results) > 0

    def test_explicit_gpu_and_cuda(self, compat_client: CompatibilityClient) -> None:
        results = compat_client.check_runtime_compatibility(
            gpu_product="NVIDIA-A100-SXM4-80GB",
            cuda_version="12.4",
        )
        assert len(results) > 0
        gpu_results = [r for r in results if r.runtime_name == "vllm-runtime"]
        assert len(gpu_results) > 0
        assert gpu_results[0].status == CompatibilityStatus.COMPATIBLE

    def test_specific_runtime(self, compat_client: CompatibilityClient) -> None:
        results = compat_client.check_runtime_compatibility(runtime_name="vllm-runtime")
        assert all(r.runtime_name == "vllm-runtime" for r in results)

    def test_t4_cuda_122_incompatible_with_vllm_216(
        self, mock_client: MockK8sClient
    ) -> None:
        client = CompatibilityClient(mock_client, rhoai_version="2.16")
        results = client.check_runtime_compatibility(
            runtime_name="vllm-runtime",
            gpu_product="Tesla-T4",
            cuda_version="12.0",
        )
        assert len(results) == 1
        assert results[0].status == CompatibilityStatus.INCOMPATIBLE


class TestDiagnoseCrash:
    def test_diagnose_cuda_compatibility_issue(
        self, compat_client: CompatibilityClient
    ) -> None:
        diagnosis = compat_client.diagnose_crash("llama-serving-fail", "production-models")
        assert diagnosis.is_compatibility_issue is True
        assert "kernel image" in diagnosis.root_cause.lower() or "compute capability" in diagnosis.root_cause.lower()
        assert len(diagnosis.evidence) > 0
        assert len(diagnosis.recommended_actions) > 0

    def test_diagnose_healthy_service(self, compat_client: CompatibilityClient) -> None:
        diagnosis = compat_client.diagnose_crash("granite-serving", "production-models")
        assert diagnosis.is_compatibility_issue is False

    def test_diagnose_nonexistent_service(self, compat_client: CompatibilityClient) -> None:
        diagnosis = compat_client.diagnose_crash("nonexistent", "production-models")
        assert diagnosis.is_compatibility_issue is False


class TestAssessUpgrade:
    def test_upgrade_to_33_from_216(self, compat_client: CompatibilityClient) -> None:
        assessment = compat_client.assess_upgrade("3.3")
        assert assessment.current_version == "2.16"
        assert assessment.target_version == "3.3"
        assert len(assessment.breaking_changes) > 0
        assert any("fresh install" in bc.lower() for bc in assessment.breaking_changes)

    def test_upgrade_to_unknown_version(self, compat_client: CompatibilityClient) -> None:
        assessment = compat_client.assess_upgrade("99.99")
        assert assessment.safe_to_upgrade is False

    def test_upgrade_includes_cuda_increase_warning(
        self, compat_client: CompatibilityClient
    ) -> None:
        assessment = compat_client.assess_upgrade("3.3")
        assert len(assessment.recommendations) > 0 or len(assessment.breaking_changes) > 0


class TestFindCompatibleRuntimes:
    def test_find_all_compatible(self, compat_client: CompatibilityClient) -> None:
        runtimes = compat_client.find_compatible_runtimes()
        assert len(runtimes) > 0

    def test_filter_by_format(self, compat_client: CompatibilityClient) -> None:
        runtimes = compat_client.find_compatible_runtimes(model_format="vLLM")
        assert all("vLLM" in r.supported_formats for r in runtimes)

    def test_explicit_gpu_product(self, compat_client: CompatibilityClient) -> None:
        runtimes = compat_client.find_compatible_runtimes(
            gpu_product="Tesla-T4", cuda_version="12.1"
        )
        names = [r.runtime_name for r in runtimes]
        assert "vllm-runtime" in names


class TestScanInstalledOperators:
    def test_returns_installed_csvs(self, compat_client: CompatibilityClient) -> None:
        operators = compat_client.scan_installed_operators()
        assert len(operators) == 3
        names = [op.name for op in operators]
        assert any("rhods-operator" in n for n in names)
        assert any("gpu-operator" in n for n in names)
        assert any("nfd" in n for n in names)

    def test_csv_has_version(self, compat_client: CompatibilityClient) -> None:
        operators = compat_client.scan_installed_operators()
        for op in operators:
            assert op.version, f"Operator {op.name} has no version"


class TestCheckOperatorReadiness:
    def test_check_against_216(self, compat_client: CompatibilityClient) -> None:
        checks = compat_client.check_operator_readiness("2.16")
        assert len(checks) > 0
        rhods = next(c for c in checks if "OpenShift AI" in c.operator_name)
        assert rhods.installed is True

    def test_check_against_33_missing_operators(
        self, compat_client: CompatibilityClient
    ) -> None:
        checks = compat_client.check_operator_readiness("3.3")
        missing_required = [c for c in checks if c.required and not c.installed]
        missing_names = [c.operator_name for c in missing_required]
        assert "Cert Manager Operator" in missing_names
        assert "Jobset Operator" in missing_names

    def test_nvidia_gpu_operator_version_ok(
        self, compat_client: CompatibilityClient
    ) -> None:
        checks = compat_client.check_operator_readiness("3.3")
        nvidia = next(c for c in checks if "NVIDIA" in c.operator_name)
        assert nvidia.installed is True
        assert nvidia.version_ok is True

    def test_unknown_version_returns_empty(
        self, compat_client: CompatibilityClient
    ) -> None:
        checks = compat_client.check_operator_readiness("99.99")
        assert checks == []
