# Runtime Compatibility Tools -- Test Questions

Start the server in mock mode before testing:

```bash
RHOAI_MCP_MOCK_CLUSTER=true uv run rhoai-mcp --transport sse
```

The mock cluster has:
- 2x A100 nodes (gpu-node-1, gpu-node-2): CUDA 12.4, driver 535.129.03, compute 8.0
- 1x T4 node (gpu-node-3): CUDA 12.2, driver 525.60.13, compute 7.5
- `granite-serving` InferenceService (healthy, on gpu-node-1)
- `llama-serving-fail` InferenceService (CrashLoopBackOff, on gpu-node-3)
- RHOAI version defaults to 2.16

---

## Tool 1: check_runtime_compatibility

### Basic cluster scan
- "Check if our cluster hardware is compatible with all serving runtimes"
- "Is our cluster compatible with the vLLM runtime?"
- "Are there any runtime compatibility issues on our cluster?"

### Specific hardware queries
- "Our hardware only supports CUDA 12.2. Which vLLM runtime should I use for RHOAI 2.16?"
- "Is the vLLM runtime compatible with Tesla T4 GPUs running CUDA 12.2?"
- "Can I run vLLM on an A100 with CUDA 12.4 on RHOAI 2.16?"
- "Check if the caikit-tgis-runtime works with our T4 nodes"

### Version-specific checks
- "Is the vLLM runtime in RHOAI 3.3 compatible with our Tesla T4 GPUs?"
- "Check runtime compatibility for RHOAI 3.3 on A100 hardware with CUDA 12.8"
- "Which runtimes in RHOAI 2.25 are compatible with CUDA 12.4?"

### Edge cases
- "Is OpenVINO compatible with our GPU hardware?" (should say it's CPU-based, no GPU needed)
- "Check compatibility for a runtime called 'nonexistent-runtime'" (should handle gracefully)

---

## Tool 2: diagnose_compatibility_issue

### Failing deployment
- "The llama-serving-fail deployment in production-models is crashing. Can you diagnose why?"
- "My model deployment llama-serving-fail in production-models is stuck in CrashLoopBackOff. Is it a CUDA compatibility issue?"
- "Diagnose why the llama-serving-fail inference service is failing"

### Healthy deployment
- "Is the granite-serving deployment in production-models having any compatibility issues?"
- "Check if granite-serving has any CUDA problems"

### Nonexistent service
- "Diagnose compatibility issues for a service called 'does-not-exist' in production-models"

---

## Tool 3: assess_upgrade_compatibility

### Upgrade from 2.16 to 3.3
- "We're planning to upgrade from RHOAI 2.16 to 3.3. Will our current model deployments still work?"
- "Assess what would break if we upgrade to RHOAI 3.3"
- "Is it safe to upgrade to RHOAI 3.3 with our current GPU hardware?"

### Upgrade path validation
- "Can we upgrade directly from RHOAI 2.16 to 3.3?" (should warn about no direct 2.x->3.x upgrade)

### Platform requirements
- "What OpenShift version do we need for RHOAI 3.3?"
- "What operators are required for RHOAI 3.3?"

### Unknown target
- "Assess compatibility for upgrading to RHOAI version 99.99"

---

## Tool 4: find_compatible_runtimes

### General discovery
- "What serving runtimes are compatible with our cluster hardware?"
- "List all runtimes that work with our GPUs"

### Hardware-specific
- "Which runtimes work with Tesla T4 GPUs and CUDA 12.2?"
- "Find runtimes compatible with A100 hardware"

### Format-filtered
- "Which runtimes support the vLLM model format on our hardware?"
- "Find a compatible runtime for serving caikit models"

### After a failure
- "The vLLM runtime isn't working on our T4 nodes. What alternatives do I have?"

---

## Tool 5: get_cluster_gpu_info

### Cluster overview
- "What GPU hardware is in our cluster?"
- "Show me the CUDA and driver versions on all GPU nodes"
- "Do we have a heterogeneous GPU cluster?"

### Pre-check before deployment
- "What GPUs and CUDA versions are available before I deploy a model?"
- "What's the compute capability of our GPU nodes?"

---

## Multi-tool scenarios

These questions should trigger the agent to chain multiple tools together:

1. "I want to deploy a vLLM model but my deployment keeps crashing on gpu-node-3. What's wrong and what should I do instead?"
   - Expected: diagnose_compatibility_issue -> find_compatible_runtimes

2. "We're upgrading to RHOAI 3.3 next month. Give me a full readiness report -- what hardware do we have, what will break, and what do we need to change?"
   - Expected: get_cluster_gpu_info -> assess_upgrade_compatibility

3. "Our T4 nodes have CUDA 12.2. Can we use vLLM on RHOAI 3.3, and if not, what are our options?"
   - Expected: check_runtime_compatibility -> find_compatible_runtimes

4. "Before I deploy, check if our cluster can run the vLLM runtime, and if any nodes will have issues"
   - Expected: get_cluster_gpu_info -> check_runtime_compatibility
