"""Tests for runtime compatibility MCP tools."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.composites.compatibility.tools import register_tools
from rhoai_mcp.mock_k8s.cluster_state import create_default_cluster_state
from rhoai_mcp.mock_k8s.mock_client import MockK8sClient


def _make_mock_mcp() -> MagicMock:
    """Create a mock FastMCP that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


def _make_mock_server() -> MagicMock:
    """Create a mock RHOAIServer with mock K8s client."""
    state = create_default_cluster_state()
    k8s = MockK8sClient(state=state)
    k8s.connect()

    server = MagicMock()
    server.k8s = k8s
    server.config.rhoai_version = "2.16"
    return server


@pytest.fixture
def tools() -> dict:
    mcp = _make_mock_mcp()
    server = _make_mock_server()
    register_tools(mcp, server)
    return mcp._registered_tools


class TestToolRegistration:
    def test_all_tools_registered(self, tools: dict) -> None:
        expected = {
            "check_runtime_compatibility",
            "diagnose_compatibility_issue",
            "assess_upgrade_compatibility",
            "find_compatible_runtimes",
            "get_cluster_gpu_info",
            "check_operator_readiness",
        }
        assert set(tools.keys()) == expected


class TestCheckRuntimeCompatibility:
    def test_auto_detect(self, tools: dict) -> None:
        result = tools["check_runtime_compatibility"]()
        assert "results" in result
        assert "summary" in result
        assert result["rhoai_version"] == "2.16"

    def test_explicit_params(self, tools: dict) -> None:
        result = tools["check_runtime_compatibility"](
            gpu_product="Tesla-T4",
            cuda_version="12.1",
        )
        assert result["summary"]["total_checks"] > 0

    def test_specific_runtime(self, tools: dict) -> None:
        result = tools["check_runtime_compatibility"](runtime_name="vllm-runtime")
        for r in result["results"]:
            assert r["runtime_name"] == "vllm-runtime"


class TestDiagnoseCompatibilityIssue:
    def test_failing_service(self, tools: dict) -> None:
        result = tools["diagnose_compatibility_issue"](
            service_name="llama-serving-fail",
            namespace="production-models",
        )
        assert result["is_compatibility_issue"] is True
        assert len(result["evidence"]) > 0
        assert len(result["recommended_actions"]) > 0

    def test_healthy_service(self, tools: dict) -> None:
        result = tools["diagnose_compatibility_issue"](
            service_name="granite-serving",
            namespace="production-models",
        )
        assert result["is_compatibility_issue"] is False


class TestAssessUpgradeCompatibility:
    def test_upgrade_to_33(self, tools: dict) -> None:
        result = tools["assess_upgrade_compatibility"](target_version="3.3")
        assert result["current_version"] == "2.16"
        assert result["target_version"] == "3.3"
        assert "summary" in result
        assert isinstance(result["safe_to_upgrade"], bool)
        assert result["target_platform_requirements"] is not None
        assert "4.19" in result["target_platform_requirements"]["openshift_versions"]
        assert len(result["target_platform_requirements"]["required_operators"]) >= 3
        assert "operator_readiness" in result
        assert "operators_ready" in result["summary"]

    def test_upgrade_shows_missing_operators(self, tools: dict) -> None:
        result = tools["assess_upgrade_compatibility"](target_version="3.3")
        assert result["missing_operators"] is not None
        missing_names = [op["name"] for op in result["missing_operators"]]
        assert "Cert Manager Operator" in missing_names

    def test_unknown_target(self, tools: dict) -> None:
        result = tools["assess_upgrade_compatibility"](target_version="99.99")
        assert result["safe_to_upgrade"] is False


class TestFindCompatibleRuntimes:
    def test_find_all(self, tools: dict) -> None:
        result = tools["find_compatible_runtimes"]()
        assert result["total"] > 0
        assert result["rhoai_version"] == "2.16"

    def test_filter_by_format(self, tools: dict) -> None:
        result = tools["find_compatible_runtimes"](model_format="vLLM")
        for rt in result["compatible_runtimes"]:
            assert "vLLM" in rt["supported_formats"]


class TestCheckOperatorReadiness:
    def test_check_current_version(self, tools: dict) -> None:
        result = tools["check_operator_readiness"]()
        assert result["rhoai_version"] == "2.16"
        assert "summary" in result
        assert "operators" in result

    def test_check_33_shows_missing(self, tools: dict) -> None:
        result = tools["check_operator_readiness"](target_version="3.3")
        assert result["summary"]["missing_required"] > 0
        assert result["summary"]["ready"] is False
        assert result["missing_required"] is not None

    def test_operator_entries_have_features(self, tools: dict) -> None:
        result = tools["check_operator_readiness"](target_version="3.3")
        for op in result["operators"]:
            assert "feature" in op
            assert "installed" in op


class TestGetClusterGPUInfo:
    def test_returns_nodes(self, tools: dict) -> None:
        result = tools["get_cluster_gpu_info"]()
        assert result["total_gpu_nodes"] == 3
        assert result["total_gpus"] == 10
        assert result["heterogeneous"] is True
        assert len(result["unique_gpu_products"]) == 2
