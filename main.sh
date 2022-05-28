#!/usr/bin/env bash
export cores=16
export i=0
export pids=""

for instance in comp_instance_1_20220110 comp_instance_2_20211007 comp_instance_3_20211215
do
    for run in $(seq 0 1)
    do
        # Checkout the branch and start a run in the background
        echo "Starting run $run on instance $instance on branch $1"
        #git checkout $branch
        #sleep 5
        python main.py --run $run --instance $instance --branch $1 &
        #sleep 5
        echo "Finished initializing run $run on instance $instance on branch $1"

        # Add the latest subprocess to the list of PIDs
        pids="$pids $!"
        i=$((i + 1))
        # Wait for all of the runs to complete before starting a new set.
        # This is to avoid starting more processes than there are cores. Could've done this better
        if [ $i -eq $cores ]; then
            echo "Waiting for tasks to finish before starting new"
            wait $pids
            i=$((0))
            pids=""
        fi
    done
done

# Wait for all remaining jobs to finish.
echo "Waiting for last set of tasks to finish"
wait $pids