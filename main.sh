#!/usr/bin/env bash
export cores=16
export i=0
export pids=""
export stacks=("requests_comp_instance_1" "requests_comp_instance_2" "requests_comp_instance_3")
export instances=("comp_instance_1_20220110" "comp_instance_2_20211007" "comp_instance_3_20211215")

for index in $(seq 1);
do
    instance=${instances[$index]}
    req_stack=${stacks[$index]}
    for run in $(seq 5)
    do
        # Checkout the branch and start a run in the background
        echo "Starting run $run on instance $instance on branch $1"
        echo "Req stack is set to $req_stack"
        #git checkout $branch
        #sleep 5
        python main.py --run $run --instance $instance --req_stack $req_stack --branch $1 &
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