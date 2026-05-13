"""Pydantic models for runtime compatibility results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CompatibilityStatus(str, Enum):
    """Overall compatibility verdict."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class GPUNodeInfo(BaseModel):
    """GPU and driver information for a single cluster node."""

    node_name: str
    gpu_product: str | None = None
    gpu_count: int = 0
    gpu_architecture: str | None = None
    compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None


class CompatibilityCheckResult(BaseModel):
    """Result of checking one runtime against one GPU/CUDA combination."""

    runtime_name: str
    runtime_display_name: str | None = None
    gpu_product: str | None = None
    gpu_architecture: str | None = None
    compute_capability: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    status: CompatibilityStatus = CompatibilityStatus.UNKNOWN
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CrashDiagnosis(BaseModel):
    """Diagnosis of a failing InferenceService."""

    service_name: str
    namespace: str
    is_compatibility_issue: bool = False
    root_cause: str = ""
    evidence: list[str] = Field(default_factory=list)
    compatibility_details: CompatibilityCheckResult | None = None
    recommended_actions: list[str] = Field(default_factory=list)


class AffectedService(BaseModel):
    """An InferenceService that may be affected by an upgrade."""

    name: str
    namespace: str
    runtime: str | None = None
    current_status: str = "Unknown"
    impact: str = ""
    severity: str = "low"


class UpgradeAssessment(BaseModel):
    """Pre-upgrade compatibility assessment."""

    current_version: str
    target_version: str
    affected_services: list[AffectedService] = Field(default_factory=list)
    safe_to_upgrade: bool = True
    breaking_changes: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class RuntimeAlternative(BaseModel):
    """A compatible runtime alternative."""

    runtime_name: str
    display_name: str | None = None
    compatibility_status: CompatibilityStatus = CompatibilityStatus.COMPATIBLE
    migration_notes: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=list)


class InstalledOperator(BaseModel):
    """An operator installed on the cluster (from ClusterServiceVersion)."""

    name: str
    namespace: str
    version: str
    phase: str


class OperatorReadiness(BaseModel):
    """Result of checking required operators against installed ones."""

    operator_name: str
    required: bool = True
    feature: str = ""
    installed: bool = False
    installed_version: str | None = None
    min_version: str | None = None
    version_ok: bool = True
    notes: str = ""
