#!/usr/bin/env python3
"""
Test script to demonstrate the PIV Runner system.

This script shows how to start a PIV job via the Flask API,
monitor its progress, and retrieve results.
"""
import time
import requests
from datetime import datetime


def test_piv_runner(base_url="http://localhost:5000"):
    """
    Test the PIV runner by starting a job and monitoring it.
    
    Parameters
    ----------
    base_url : str
        Base URL of the Flask server (default: http://localhost:5000)
    """
    print("=" * 70)
    print("PIV Runner Test")
    print("=" * 70)
    
    # 1. Start a PIV job
    print("\n[1] Starting PIV job...")
    print("    (Using settings from config.yaml)")
    start_payload = {}  # No parameters needed - uses config.yaml
    
    try:
        response = requests.post(
            f"{base_url}/run_piv",
            json=start_payload,
            timeout=10
        )
        response.raise_for_status()
        job = response.json()
        
        if job.get("status") != "started":
            print(f"❌ Failed to start job: {job}")
            return
        
        job_id = job["job_id"]
        print(f"✅ Job started successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   PID: {job['pid']}")
        print(f"   Log file: {job['log_file']}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server.")
        print("   Make sure the server is running: python src/server.py")
        return
    except Exception as e:
        print(f"❌ Error starting job: {e}")
        return
    
    # 2. Monitor job progress
    print(f"\n[2] Monitoring job progress...")
    print("    (Press Ctrl+C to cancel and exit)")
    
    try:
        last_log_count = 0
        while True:
            time.sleep(3)  # Check every 3 seconds
            
            # Get job status
            status_response = requests.get(
                f"{base_url}/piv_status",
                params={"job_id": job_id},
                timeout=10
            )
            status_response.raise_for_status()
            status = status_response.json()
            
            # Display status
            elapsed = status.get("elapsed_seconds", 0)
            running = status.get("running", False)
            
            if running:
                print(f"    ⏳ Running... {elapsed:.1f}s elapsed")
                
                # Show new log lines if available
                log_tail = status.get("log_tail", [])
                if len(log_tail) > last_log_count:
                    new_lines = log_tail[last_log_count:]
                    for line in new_lines[-3:]:  # Show last 3 new lines
                        print(f"       📝 {line.strip()}")
                    last_log_count = len(log_tail)
            else:
                # Job finished
                return_code = status.get("return_code", -1)
                end_time = status.get("end_time")
                
                print(f"\n    ✅ Job completed!")
                print(f"       Return code: {return_code}")
                print(f"       Total time: {elapsed:.1f}s")
                print(f"       End time: {end_time}")
                
                if return_code == 0:
                    print(f"\n✨ PIV computation successful!")
                else:
                    print(f"\n⚠️  Job finished with non-zero exit code: {return_code}")
                    print(f"    Check log file for details: {status.get('log_file')}")
                
                break
                
    except KeyboardInterrupt:
        print(f"\n\n[3] Cancelling job...")
        
        # Cancel the job
        cancel_response = requests.post(
            f"{base_url}/cancel_piv",
            json={"job_id": job_id},
            timeout=10
        )
        
        if cancel_response.status_code == 200:
            print(f"    ✅ Job cancelled successfully")
        else:
            print(f"    ⚠️  Failed to cancel job: {cancel_response.json()}")
    
    except Exception as e:
        print(f"\n❌ Error monitoring job: {e}")
    
    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)


def test_list_jobs(base_url="http://localhost:5000"):
    """List all PIV jobs."""
    print("\n[List All Jobs]")
    try:
        response = requests.get(f"{base_url}/piv_status", timeout=10)
        response.raise_for_status()
        data = response.json()
        jobs = data.get("jobs", [])
        
        if not jobs:
            print("  No jobs found.")
        else:
            print(f"  Found {len(jobs)} job(s):")
            for job in jobs:
                status = "🟢 Running" if job.get("running") else "🔴 Finished"
                elapsed = job.get("elapsed_seconds", 0)
                print(f"    - {job['job_id']}: {status} ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  ❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if server is reachable first
    print("Checking Flask server connection...")
    try:
        requests.get("http://localhost:5000/config", timeout=2)
        print("✅ Server is running\n")
    except:
        print("❌ Flask server is not running or not reachable.")
        print("   Start it with: python src/server.py")
        sys.exit(1)
    
    # Run the test
    test_piv_runner()
    
    # List all jobs
    test_list_jobs()
