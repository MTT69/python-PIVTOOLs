# PIV Runner Usage Guide

## Overview

The PIV Runner system allows the Flask server to launch full PIV computations as separate subprocesses. This approach provides:

✅ **Full computational performance** - No GIL limitations, full CPU/GPU access  
✅ **Non-blocking server** - Flask remains responsive during PIV processing  
✅ **Process isolation** - PIV crashes don't affect the server  
✅ **Job tracking** - Monitor progress, check logs, cancel jobs  
✅ **Resource management** - Each PIV job gets its own Dask cluster  

## Architecture

```
Flask Server (port 5000)
    │
    ├─ /run_piv endpoint
    │     └─ spawns subprocess → python pypivtools/example.py
    │                                  │
    │                                  ├─ Starts Dask cluster
    │                                  ├─ Loads images
    │                                  ├─ Performs PIV
    │                                  └─ Saves results
    │
    ├─ /piv_status endpoint (check progress)
    └─ /cancel_piv endpoint (terminate job)
```

## API Endpoints

### 1. Start PIV Job

```bash
POST /run_piv
Content-Type: application/json

{}  # No parameters needed - uses config.yaml settings
```

**Note:** Currently, the PIV job runs with all settings from `config.yaml`. The API accepts optional parameters for future compatibility, but they are not yet implemented.

**Response:**
```json
{
  "status": "started",
  "job_id": "piv_20231005_143022",
  "pid": 12345,
  "log_file": "/path/to/logs/piv_runs/piv_20231005_143022.log"
}
```

### 2. Check PIV Status

```bash
# Get specific job status
GET /piv_status?job_id=piv_20231005_143022

# Get all jobs
GET /piv_status
```

**Response:**
```json
{
  "job_id": "piv_20231005_143022",
  "running": true,
  "start_time": "2023-10-05T14:30:22",
  "end_time": null,
  "elapsed_seconds": 125.3,
  "return_code": null,
  "log_file": "/path/to/logs/piv_runs/piv_20231005_143022.log",
  "log_tail": ["INFO: Processing frame 45/100", "..."]
}
```

### 3. Cancel PIV Job

```bash
POST /cancel_piv
Content-Type: application/json

{
  "job_id": "piv_20231005_143022"
}
```

**Response:**
```json
{
  "status": "cancelled",
  "job_id": "piv_20231005_143022"
}
```

## Usage Examples

### Python/Requests

```python
import requests
import time

# Start PIV job (uses config.yaml settings)
response = requests.post('http://localhost:5000/run_piv', json={})
job = response.json()
job_id = job['job_id']
print(f"Started job: {job_id}")

# Poll status
while True:
    status = requests.get(f'http://localhost:5000/piv_status?job_id={job_id}').json()
    
    if not status['running']:
        print(f"Job completed with return code: {status['return_code']}")
        break
    
    print(f"Running... {status['elapsed_seconds']:.1f}s elapsed")
    print(f"Recent log: {status['log_tail'][-1] if status['log_tail'] else 'N/A'}")
    time.sleep(5)

# Cancel if needed
# requests.post('http://localhost:5000/cancel_piv', json={'job_id': job_id})
```

### cURL

```bash
# Start job (uses config.yaml settings)
curl -X POST http://localhost:5000/run_piv \
  -H "Content-Type: application/json" \
  -d '{}'

# Check status
curl "http://localhost:5000/piv_status?job_id=piv_20231005_143022"

# Cancel job
curl -X POST http://localhost:5000/cancel_piv \
  -H "Content-Type: application/json" \
  -d '{"job_id": "piv_20231005_143022"}'
```

### JavaScript/Fetch

```javascript
// Start PIV job (uses config.yaml settings)
const response = await fetch('http://localhost:5000/run_piv', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({})
});
const job = await response.json();
console.log('Started job:', job.job_id);

// Poll status
const checkStatus = async (jobId) => {
  const status = await fetch(`http://localhost:5000/piv_status?job_id=${jobId}`)
    .then(r => r.json());
  
  if (status.running) {
    console.log(`Running... ${status.elapsed_seconds}s`);
    setTimeout(() => checkStatus(jobId), 5000);
  } else {
    console.log('Job completed:', status.return_code);
  }
};

checkStatus(job.job_id);
```

## Log Files

All PIV job logs are stored in:
```
logs/piv_runs/piv_YYYYMMDD_HHMMSS.log
```

You can tail these logs in real-time:
```bash
tail -f logs/piv_runs/piv_20231005_143022.log
```

## Benefits Over In-Flask Execution

| Aspect | In-Flask (old) | Subprocess (new) |
|--------|----------------|------------------|
| **Computational Resources** | Limited by GIL | Full access |
| **Server Responsiveness** | Blocked | Non-blocking |
| **Memory Isolation** | Shared | Isolated |
| **Crash Impact** | Kills server | Isolated |
| **Dask Cluster** | Conflicts with Flask | Dedicated |
| **Monitoring** | Difficult | Built-in logs |
| **Cancellation** | N/A | Clean termination |

## Current Behavior

The PIV runner currently executes `example.py` which reads all settings from `config.yaml`. To change which cameras or paths to process, update `config.yaml` before starting a job.

## Future Enhancements

Future enhancements could include:

1. **CLI Arguments**: Pass `--cameras`, `--source-path-idx` via command line
2. **Config Overrides**: Apply temporary config changes via environment variables or API
3. **Progress Reporting**: Structured JSON progress updates (frame count, % complete)
4. **Result Streaming**: Stream results back to server as they complete
5. **Queue System**: Queue multiple jobs with priority

## Troubleshooting

### Job won't start
- Check Python environment: `logs/piv_runs/*.log`
- Verify `pypivtools/example.py` exists
- Ensure virtual environment is activated: `piv/bin/python`

### Job stuck running
- Check process: `ps aux | grep example.py`
- Force kill: `kill -9 <PID>`
- Clean up: Restart Flask server

### No log output
- Ensure `logs/piv_runs/` directory exists
- Check file permissions
- Verify log file path in job status response

## Architecture Notes

The `PIVRunner` class in `src/piv_runner.py` manages:
- Subprocess spawning with proper environment
- Log file creation and rotation
- Job tracking and status monitoring
- Graceful cancellation (SIGTERM → SIGKILL)
- Automatic cleanup of finished jobs

The design keeps the server lightweight while giving PIV computations full system resources.
