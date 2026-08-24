#!/bin/bash
# Exit immediately if a command fails.
set -e

# Allowed values: "current", "delta"
# current: use the boltz command from the environment that is already active.
# delta:   use the Boltz environment installed during the Delta tests.
server="current"

echo "Server: $server"

usage() {
    echo ""
    echo "Please make sure all required parameters are given"
    echo "Usage: $0 <OPTIONS>"
    echo "Required Parameters:"
    echo "-i <input_path>                 Input YAML/FASTA file or a directory of inputs."
    echo "-o <output_dir>                 Directory in which results will be saved."
    echo "Optional Parameters:"
    echo "-d <gpu_device>                 CUDA device IDs, for example 0 or 0,1. (default: 0)"
    echo "-D <run_data_pipeline>          Run the data pipeline. (default: true)"
    echo "-P <run_inference>              Run model inference. (default: true)"
    echo "-r <model_seeds>                One seed or comma-separated seeds, e.g. 1,2,3."
    echo "-n <diffusion_samples>          Number of samples per seed. (default: 1)"
    echo "-c <recycling_steps>            Number of recycling steps. (default: 3)"
    echo "-p <sampling_steps>             Number of diffusion sampling steps. (default: 200)"
    echo "-m <max_parallel_samples>       Samples evaluated in parallel. (default: 5)"
    echo "-M <use_msa_server>             Search missing protein MSAs remotely. (default: true)"
    echo "-S <skip>                       Skip seeds whose expected outputs exist. (default: false)"
    echo "-h                              Show this help."
    echo ""
    echo "Examples:"
    echo "  # Run only the data pipeline."
    echo "  $0 -i seq.yaml -o result -D true -P false"
    echo ""
    echo "  # Read seq_data.yaml and predict five seeds."
    echo "  $0 -i result/seq/seq_data.yaml -o result -D false -P true -r 1,2,3,4,5 -n 5 -S true"
    exit 1
}

# region: Parse command line arguments
while getopts "i:o:d:D:P:r:n:c:p:m:M:S:h" opt; do
    case "${opt}" in
    i) input_path=$OPTARG ;;
    o) output_dir=$OPTARG ;;
    d) gpu_device=$OPTARG ;;
    D) run_data_pipeline=$OPTARG ;;
    P) run_inference=$OPTARG ;;
    r) model_seeds=$OPTARG ;;
    n) diffusion_samples=$OPTARG ;;
    c) recycling_steps=$OPTARG ;;
    p) sampling_steps=$OPTARG ;;
    m) max_parallel_samples=$OPTARG ;;
    M) use_msa_server=$OPTARG ;;
    S) skip=$OPTARG ;;
    h) usage ;;
    *) usage ;;
    esac
done
# endregion

# region: Check required parameters
if [[ "$input_path" == "" || "$output_dir" == "" ]]; then
    usage
fi

if [[ ! -e "$input_path" ]]; then
    echo "Error: input path does not exist: $input_path"
    exit 1
fi
# endregion

# region: Set default values
if [[ "$gpu_device" == "" ]]; then gpu_device="0"; fi
if [[ "$run_data_pipeline" == "" ]]; then run_data_pipeline="true"; fi
if [[ "$run_inference" == "" ]]; then run_inference="true"; fi
if [[ "$diffusion_samples" == "" ]]; then diffusion_samples="1"; fi
if [[ "$recycling_steps" == "" ]]; then recycling_steps="3"; fi
if [[ "$sampling_steps" == "" ]]; then sampling_steps="200"; fi
if [[ "$max_parallel_samples" == "" ]]; then max_parallel_samples="5"; fi
if [[ "$use_msa_server" == "" ]]; then use_msa_server="true"; fi
if [[ "$skip" == "" ]]; then skip="false"; fi

if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    echo "Error: run_data_pipeline and run_inference cannot both be false."
    exit 1
fi
# endregion

# region: Set paths and activate the environment for each server
if [[ "$server" == "current" ]]; then
    # Activate your environment before running this script.
    boltz_bin="boltz"
    cache_dir=""

elif [[ "$server" == "delta" ]]; then
    env_path=/work/hdd/bbgs/nwentao/boltz2_wrapper_test/venv/bin/activate
    boltz_bin=/work/hdd/bbgs/nwentao/boltz2_wrapper_test/venv/bin/boltz
    cache_dir=/work/hdd/bbgs/nwentao/boltz2_wrapper_test/cache/boltz

    source "$env_path"
else
    echo "Error: server is invalid: $server"
    exit 1
fi
# endregion

if ! command -v "$boltz_bin" >/dev/null 2>&1; then
    echo "Error: Boltz executable does not exist: $boltz_bin"
    exit 1
fi

if [[ "$run_inference" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_device"
    IFS=',' read -r -a gpu_device_list <<< "$gpu_device"
    devices=${#gpu_device_list[@]}
else
    devices=1
fi

#### Command arguments
command_args=(
    predict "$input_path"
    --out_dir "$output_dir"
    --run_data_pipeline "$run_data_pipeline"
    --run_inference "$run_inference"
    --devices "$devices"
    --recycling_steps "$recycling_steps"
    --sampling_steps "$sampling_steps"
    --diffusion_samples "$diffusion_samples"
    --max_parallel_samples "$max_parallel_samples"
)

if [[ "$cache_dir" != "" ]]; then
    command_args+=(--cache "$cache_dir")
fi

if [[ "$model_seeds" != "" ]]; then
    command_args+=(--seeds "$model_seeds")
fi

if [[ "$use_msa_server" == "true" ]]; then
    command_args+=(--use_msa_server)
fi

if [[ "$skip" == "true" ]]; then
    command_args+=(--skip)
fi

# Run Boltz with the requested parameters.
echo "$boltz_bin ${command_args[*]}"
"$boltz_bin" "${command_args[@]}"
