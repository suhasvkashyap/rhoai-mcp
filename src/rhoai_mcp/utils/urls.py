"""URL builder utilities for generating console and dashboard links."""

from urllib.parse import quote


class SourceURLBuilder:
    """Builds console and dashboard URLs for Kubernetes resources.

    OpenShift Console URL patterns:
    - Projects: /k8s/cluster/projects/{name}
    - Namespaced CRDs: /k8s/ns/{namespace}/{group}~{version}~{kind}/{name}
    - Core resources: /k8s/ns/{namespace}/{resource-type}/{name}

    RHOAI Dashboard URL patterns:
    - Projects: /projects/{namespace}
    - Workbenches: /projects/{namespace}?section=workbenches
    - Models: /projects/{namespace}/modelServing
    - Data connections: /projects/{namespace}?section=data-connections
    - Cluster storage: /projects/{namespace}?section=cluster-storage
    """

    # Map kind to OpenShift console resource path
    CONSOLE_PATHS: dict[str, str] = {
        "Project": "/k8s/cluster/projects/{name}",
        "Notebook": "/k8s/ns/{namespace}/kubeflow.org~v1~Notebook/{name}",
        "InferenceService": "/k8s/ns/{namespace}/serving.kserve.io~v1beta1~InferenceService/{name}",
        "Secret": "/k8s/ns/{namespace}/secrets/{name}",
        "PersistentVolumeClaim": "/k8s/ns/{namespace}/persistentvolumeclaims/{name}",
        "DataSciencePipelinesApplication": (
            "/k8s/ns/{namespace}/"
            "datasciencepipelinesapplications.opendatahub.io~v1alpha1~DataSciencePipelinesApplication/{name}"
        ),
        "TrainJob": "/k8s/ns/{namespace}/trainer.kubeflow.org~v1~TrainJob/{name}",
    }

    # Map kind to RHOAI dashboard path
    DASHBOARD_PATHS: dict[str, str] = {
        "Project": "/projects/{namespace}",
        "Notebook": "/projects/{namespace}?section=workbenches",
        "InferenceService": "/projects/{namespace}/modelServing",
        "Secret": "/projects/{namespace}?section=data-connections",
        "PersistentVolumeClaim": "/projects/{namespace}?section=cluster-storage",
        "DataSciencePipelinesApplication": "/projects/{namespace}?section=pipelines-projects",
        "TrainJob": "/projects/{namespace}",
    }

    def __init__(
        self,
        console_base_url: str | None = None,
        dashboard_base_url: str | None = None,
    ) -> None:
        """Initialize the URL builder.

        Args:
            console_base_url: OpenShift console base URL (without trailing slash).
            dashboard_base_url: RHOAI dashboard base URL (without trailing slash).
        """
        self.console_base_url = console_base_url.rstrip("/") if console_base_url else None
        self.dashboard_base_url = dashboard_base_url.rstrip("/") if dashboard_base_url else None

    def build_console_url(
        self,
        kind: str | None,
        name: str,
        namespace: str | None = None,
    ) -> str | None:
        """Build OpenShift console URL for a resource.

        Args:
            kind: Resource kind (e.g., "Notebook", "InferenceService").
            name: Resource name.
            namespace: Resource namespace (required for namespaced resources).

        Returns:
            Full console URL or None if console_base_url is not configured.
        """
        if not self.console_base_url or not kind:
            return None

        path_template = self.CONSOLE_PATHS.get(kind)
        if not path_template:
            return None

        # For Project, use name as both name and namespace
        effective_namespace = namespace if namespace else name

        path = path_template.format(
            name=quote(name, safe=""),
            namespace=quote(effective_namespace, safe="") if effective_namespace else "",
        )

        return f"{self.console_base_url}{path}"

    def build_dashboard_url(
        self,
        kind: str | None,
        name: str,
        namespace: str | None = None,
    ) -> str | None:
        """Build RHOAI dashboard URL for a resource.

        Args:
            kind: Resource kind (e.g., "Notebook", "InferenceService").
            name: Resource name.
            namespace: Resource namespace.

        Returns:
            Full dashboard URL or None if dashboard_base_url is not configured.
        """
        if not self.dashboard_base_url or not kind:
            return None

        path_template = self.DASHBOARD_PATHS.get(kind)
        if not path_template:
            return None

        # For Project, use name as the namespace
        effective_namespace = namespace if namespace else name

        path = path_template.format(
            name=quote(name, safe=""),
            namespace=quote(effective_namespace, safe="") if effective_namespace else "",
        )

        return f"{self.dashboard_base_url}{path}"


# Global URL builder instance (lazy-initialized)
_url_builder: SourceURLBuilder | None = None


def get_url_builder() -> SourceURLBuilder:
    """Get the global URL builder instance, initialized from config."""
    global _url_builder
    if _url_builder is None:
        from rhoai_mcp.config import get_config

        config = get_config()
        _url_builder = SourceURLBuilder(
            console_base_url=config.console_url,
            dashboard_base_url=config.dashboard_url,
        )
    return _url_builder


def reset_url_builder() -> None:
    """Reset the global URL builder (useful for testing)."""
    global _url_builder
    _url_builder = None
