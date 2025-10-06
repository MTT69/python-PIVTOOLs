# PIV Runner Implementation Summary

## What Was Built

A **subprocess-based PIV execution system** that allows the Flask server to spawn full PIV computations as separate processes, providing maximum computational performance while keeping the server responsive.

## Key Files Created/Modified

### 1. **`src/piv_runner.py`** (NEW) 
- `PIVProcess` class: Manages individual PIV subprocess
- `PIVRunner` class: Orchestrates job creation, tracking, and cancellation
- Features:
  - Job ID generation with timestamps
  - Log file management
  - Process monitoring and status tracking
  - Graceful cancellation (SIGTERM → SIGKILL)
  - Thread-safe job tracking

### 2. **`src/server.py`** (MODIFIED)
- **`/run_piv`** endpoint: Starts PIV job as subprocess
- **`/piv_status`** endpoint: Gets status of job(s)
- **`/cancel_piv`** endpoint: Cancels running job
- **`/cancel_run`** endpoint: Legacy compatibility

### 3. **`pypivtools/example.py`** (UNCHANGED)
- Runs exactly as before, using `config.yaml` for all settings
- No modifications needed - subprocess spawns it directly
- Full computational performance maintained

### 4. **`PIV_RUNNER_USAGE.md`** (NEW)
Complete documentation with:
- Architecture overview
- API endpoint reference
- Usage examples (Python, cURL, JavaScript)
- Benefits comparison table
- Troubleshooting guide

### 5. **`test_piv_runner.py`** (NEW)
Interactive test script demonstrating:
- Starting a PIV job
- Real-time progress monitoring
- Log tailing
- Job cancellation
- Listing all jobs

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask Server (GIL)                     │
│                                                             │
│  POST /run_piv                                              │
│       │                                                     │
│       ├──► PIVRunner.start_piv_job()                        │
│       │         │                                           │
│       │         ├──► subprocess.Popen()                     │
│       │         │         │                                 │
│       │         │         └──► python pypivtools/example.py │
│       │         │                       │                   │
│       │         └──► Returns job_id     │                   │
│       │                                 │                   │
│  GET /piv_status?job_id=xxx             │                   │
│       │                                 │                   │
│       └──► Check process.poll()         │                   │
│               Read log file             │                   │
│                                         │                   │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          │ (Separate Process)
┌─────────────────────────────────────────┼───────────────────┐
│                    PIV Computation      ▼                   │
│                                                             │
│  1. Start Dask cluster                                      │
│  2. Load images                                             │
│  3. Perform PIV (all cores/GPUs available)                  │
│  4. Save results                                            │
│  5. Exit with return code                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Benefits

### ✅ Full Computational Performance
- **No GIL limitations**: Subprocess runs outside Python's Global Interpreter Lock
- **Dedicated resources**: Full access to all CPU cores and GPUs
- **Isolated Dask cluster**: PIV gets its own distributed computing environment
- **Memory isolation**: PIV memory usage doesn't affect Flask

### ✅ Non-Blocking Server
- Flask remains **fully responsive** during PIV computation
- Can handle other API requests (masking, plotting, config updates)
- Multiple PIV jobs can be tracked simultaneously
- Server won't crash if PIV encounters errors

### ✅ Job Management
- **Unique job IDs**: Track multiple PIV runs
- **Real-time logs**: Stream progress information
- **Status monitoring**: Check if running, elapsed time, completion
- **Cancellation**: Clean job termination with SIGTERM/SIGKILL
- **History tracking**: Keep recent job records

### ✅ Production Ready
- **Robust error handling**: PIV failures don't kill server
- **Log persistence**: All runs logged to disk
- **Environment aware**: Auto-detects virtual environment
- **Thread-safe**: Concurrent job management with locks

## Usage Example

### Starting a PIV Job
```python
import requests

# Start PIV job (uses config.yaml settings)
response = requests.post('http://localhost:5000/run_piv', json={})

job = response.json()
# {'status': 'started', 'job_id': 'piv_20231005_143022', 'pid': 12345, ...}
```

### Monitoring Progress
```python
# Check status
status = requests.get(
    'http://localhost:5000/piv_status',
    params={'job_id': job['job_id']}
).json()

print(f"Running: {status['running']}")
print(f"Elapsed: {status['elapsed_seconds']}s")
print(f"Recent log: {status['log_tail'][-1]}")
```

### Cancelling a Job
```python
requests.post('http://localhost:5000/cancel_piv', json={
    'job_id': job['job_id']
})
```

## Testing

Run the test script:
```bash
# Make sure Flask server is running first
python src/server.py

# In another terminal:
python test_piv_runner.py
```

The test script will:
1. Start a PIV job
2. Monitor progress in real-time
3. Display log updates
4. Handle Ctrl+C gracefully (cancels job)
5. List all tracked jobs

## Log Files

All PIV runs are logged to:
```
logs/piv_runs/piv_YYYYMMDD_HHMMSS.log
```

You can tail logs in real-time:
```bash
tail -f logs/piv_runs/piv_20231005_143022.log
```

## Future Enhancements

### Short Term
- [ ] Structured progress updates (% completion, current frame)
- [ ] Estimated time remaining based on frame rate
- [ ] Email/webhook notifications on completion
- [ ] Cleanup old log files automatically

### Medium Term
- [ ] Job queue with priority levels
- [ ] Parallel execution of multiple independent jobs
- [ ] Result streaming back to server as frames complete
- [ ] Web UI for job management dashboard

### Long Term
- [ ] Distributed execution across multiple machines
- [ ] Resume interrupted jobs from checkpoint
- [ ] Compare results between different PIV runs
- [ ] Integration with workflow orchestration (Airflow, Prefect)

## Comparison: Old vs New

| Feature | In-Flask Execution | Subprocess Execution |
|---------|-------------------|---------------------|
| **Max Performance** | ❌ GIL limited | ✅ Full resources |
| **Server Responsive** | ❌ Blocks | ✅ Non-blocking |
| **Memory Isolation** | ❌ Shared | ✅ Isolated |
| **Crash Handling** | ❌ Kills server | ✅ Isolated |
| **Progress Tracking** | ❌ Difficult | ✅ Built-in |
| **Cancellation** | ❌ N/A | ✅ Clean SIGTERM |
| **Logging** | ⚠️ Mixed | ✅ Dedicated files |
| **Multiple Jobs** | ❌ Sequential | ✅ Trackable |

## Notes

- The implementation reuses your existing `example.py` code
- Minimal changes to existing codebase
- Backward compatible (example.py can still be run directly)
- Production-ready with robust error handling
- Scales to multiple concurrent jobs with proper tracking

## Conclusion

This implementation gives you the best of both worlds:
- **Flask**: Lightweight, responsive API server for frontend communication
- **PIV**: Full computational power in isolated subprocesses

The architecture is clean, maintainable, and ready for production use.
