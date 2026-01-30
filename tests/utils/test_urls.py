"""Tests for URL builder utilities."""

import pytest

from rhoai_mcp.utils.urls import SourceURLBuilder, reset_url_builder


class TestSourceURLBuilder:
    """Tests for SourceURLBuilder class."""

    def test_build_console_url_without_base_url(self) -> None:
        """Test that None is returned when console_base_url is not configured."""
        builder = SourceURLBuilder()
        url = builder.build_console_url("Notebook", "my-wb", "my-ns")
        assert url is None

    def test_build_dashboard_url_without_base_url(self) -> None:
        """Test that None is returned when dashboard_base_url is not configured."""
        builder = SourceURLBuilder()
        url = builder.build_dashboard_url("Notebook", "my-wb", "my-ns")
        assert url is None

    def test_build_console_url_for_notebook(self) -> None:
        """Test console URL generation for Notebook resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("Notebook", "my-workbench", "my-project")

        assert url == (
            "https://console.example.com"
            "/k8s/ns/my-project/kubeflow.org~v1~Notebook/my-workbench"
        )

    def test_build_console_url_for_project(self) -> None:
        """Test console URL generation for Project resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("Project", "ml-team", None)

        assert url == "https://console.example.com/k8s/cluster/projects/ml-team"

    def test_build_console_url_for_inference_service(self) -> None:
        """Test console URL generation for InferenceService resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("InferenceService", "my-model", "my-ns")

        assert url == (
            "https://console.example.com"
            "/k8s/ns/my-ns/serving.kserve.io~v1beta1~InferenceService/my-model"
        )

    def test_build_console_url_for_secret(self) -> None:
        """Test console URL generation for Secret resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("Secret", "aws-creds", "my-ns")

        assert url == "https://console.example.com/k8s/ns/my-ns/secrets/aws-creds"

    def test_build_console_url_for_pvc(self) -> None:
        """Test console URL generation for PVC resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("PersistentVolumeClaim", "my-storage", "my-ns")

        assert url == (
            "https://console.example.com/k8s/ns/my-ns/persistentvolumeclaims/my-storage"
        )

    def test_build_console_url_for_train_job(self) -> None:
        """Test console URL generation for TrainJob resources."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("TrainJob", "train-123", "my-ns")

        assert url == (
            "https://console.example.com"
            "/k8s/ns/my-ns/trainer.kubeflow.org~v1~TrainJob/train-123"
        )

    def test_build_console_url_unknown_kind_returns_none(self) -> None:
        """Test that unknown kinds return None."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("UnknownKind", "name", "ns")

        assert url is None

    def test_build_dashboard_url_for_notebook(self) -> None:
        """Test dashboard URL generation for Notebook resources."""
        builder = SourceURLBuilder(dashboard_base_url="https://dashboard.example.com")
        url = builder.build_dashboard_url("Notebook", "my-wb", "my-project")

        assert url == "https://dashboard.example.com/projects/my-project?section=workbenches"

    def test_build_dashboard_url_for_project(self) -> None:
        """Test dashboard URL generation for Project resources."""
        builder = SourceURLBuilder(dashboard_base_url="https://dashboard.example.com")
        url = builder.build_dashboard_url("Project", "ml-team", None)

        assert url == "https://dashboard.example.com/projects/ml-team"

    def test_build_dashboard_url_for_inference_service(self) -> None:
        """Test dashboard URL generation for InferenceService resources."""
        builder = SourceURLBuilder(dashboard_base_url="https://dashboard.example.com")
        url = builder.build_dashboard_url("InferenceService", "my-model", "my-ns")

        assert url == "https://dashboard.example.com/projects/my-ns/modelServing"

    def test_build_dashboard_url_for_data_connection(self) -> None:
        """Test dashboard URL generation for Secret (data connection) resources."""
        builder = SourceURLBuilder(dashboard_base_url="https://dashboard.example.com")
        url = builder.build_dashboard_url("Secret", "aws-creds", "my-ns")

        assert url == "https://dashboard.example.com/projects/my-ns?section=data-connections"

    def test_build_dashboard_url_for_storage(self) -> None:
        """Test dashboard URL generation for PVC resources."""
        builder = SourceURLBuilder(dashboard_base_url="https://dashboard.example.com")
        url = builder.build_dashboard_url("PersistentVolumeClaim", "my-pvc", "my-ns")

        assert url == "https://dashboard.example.com/projects/my-ns?section=cluster-storage"

    def test_url_encoding_special_characters(self) -> None:
        """Test that special characters in names are URL-encoded."""
        builder = SourceURLBuilder(console_base_url="https://console.example.com")
        url = builder.build_console_url("Secret", "my secret/name", "my-ns")

        assert url == "https://console.example.com/k8s/ns/my-ns/secrets/my%20secret%2Fname"

    def test_trailing_slash_is_stripped(self) -> None:
        """Test that trailing slashes in base URLs are handled correctly."""
        builder = SourceURLBuilder(
            console_base_url="https://console.example.com/",
            dashboard_base_url="https://dashboard.example.com/",
        )
        console_url = builder.build_console_url("Project", "test", None)
        dashboard_url = builder.build_dashboard_url("Project", "test", None)

        assert console_url == "https://console.example.com/k8s/cluster/projects/test"
        assert dashboard_url == "https://dashboard.example.com/projects/test"

    def test_both_urls_configured(self) -> None:
        """Test that both URLs can be built when both are configured."""
        builder = SourceURLBuilder(
            console_base_url="https://console.example.com",
            dashboard_base_url="https://dashboard.example.com",
        )

        console_url = builder.build_console_url("Notebook", "wb", "ns")
        dashboard_url = builder.build_dashboard_url("Notebook", "wb", "ns")

        assert console_url is not None
        assert dashboard_url is not None
        assert "console.example.com" in console_url
        assert "dashboard.example.com" in dashboard_url


@pytest.fixture(autouse=True)
def reset_global_builder() -> None:
    """Reset the global URL builder before each test."""
    reset_url_builder()
