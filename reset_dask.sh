#!/bin/bash

echo "🔄 Stopping all Dask, Python, and related processes..."

# Kill all running Dask workers, schedulers, and clients
pkill -f "dask"
pkill -f "distributed.nanny"
pkill -f "dask-worker"
pkill -f "dask-scheduler"

# Kill any running Python processes (Be careful if other Python scripts are running)
pkill -f "python"

# Kill orphaned Dask processes using ports
for port in $(seq 8787 8800); do
    lsof -ti :$port | xargs kill -9 2>/dev/null
done

# Clean up temporary files used by Dask
rm -rf /tmp/dask
rm -rf ~/.dask 

echo "✅ Dask and Python processes have been reset!"

