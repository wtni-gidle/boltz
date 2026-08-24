#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run the EnsembleFold Boltz wrapper.

Usage:
  run_boltz.sh -i INPUT -o OUTPUT [options] [-- extra boltz options]

Required:
  -i PATH   Input YAML/FASTA file or input directory.
  -o PATH   Output collection root.

Common options:
  -d IDS    CUDA device IDs, for example 0 or 0,1.                 [0]
  -D BOOL   Run the data pipeline.                                [true]
  -P BOOL   Run model inference.                                  [true]
  -r LIST   One seed or comma-separated seeds, for example 1,2,3.
  -n INT    Number of diffusion samples.                          [1]
  -c INT    Number of recycling steps.                            [3]
  -p INT    Number of diffusion sampling steps.                   [200]
  -m INT    Maximum samples evaluated in parallel.                [5]
  -a NAME   Accelerator: gpu, cpu, or tpu.                        [gpu]
  -C PATH   Boltz cache directory. Defaults to BOLTZ_CACHE/Boltz.
  -M BOOL   Use the MSA server for missing protein MSAs.           [true]
  -S BOOL   Skip seeds whose required outputs already exist.      [false]
  -h        Show this help.

Environment:
  BOLTZ_ENV_ACTIVATE  Optional path to a venv/conda activate script.
  BOLTZ_BIN           Optional path to the boltz executable.
  BOLTZ_CACHE         Standard Boltz cache override.

Examples:
  # Data only: writes <name>_data.yaml and *.a3m.zst.
  run_boltz.sh -i seq.yaml -o result -D true -P false

  # Inference only: five seeds and five samples per seed.
  run_boltz.sh -i result/seq/seq_data.yaml -o result \
    -D false -P true -r 1,2,3,4,5 -n 5 -S true

  # Pass less common native options after --.
  run_boltz.sh -i seq.yaml -o result -- --no_kernels --write_full_pae
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

normalize_bool() {
    case "$2" in
        [Tt][Rr][Uu][Ee]) printf 'true' ;;
        [Ff][Aa][Ll][Ss][Ee]) printf 'false' ;;
        *) die "$1 must be true or false, got: $2" ;;
    esac
}

input_path=""
output_dir=""
gpu_devices="0"
run_data_pipeline="true"
run_inference="true"
seeds=""
diffusion_samples="1"
recycling_steps="3"
sampling_steps="200"
max_parallel_samples="5"
accelerator="gpu"
cache_dir="${BOLTZ_CACHE:-}"
use_msa_server="true"
skip="false"

while getopts ":i:o:d:D:P:r:n:c:p:m:a:C:M:S:h" opt; do
    case "$opt" in
        i) input_path="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        d) gpu_devices="$OPTARG" ;;
        D) run_data_pipeline="$OPTARG" ;;
        P) run_inference="$OPTARG" ;;
        r) seeds="$OPTARG" ;;
        n) diffusion_samples="$OPTARG" ;;
        c) recycling_steps="$OPTARG" ;;
        p) sampling_steps="$OPTARG" ;;
        m) max_parallel_samples="$OPTARG" ;;
        a) accelerator="$OPTARG" ;;
        C) cache_dir="$OPTARG" ;;
        M) use_msa_server="$OPTARG" ;;
        S) skip="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) die "Option -$OPTARG requires a value." ;;
        \?) die "Unknown option: -$OPTARG" ;;
    esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then
    shift
fi

[[ -n "$input_path" ]] || { usage >&2; die "-i is required."; }
[[ -n "$output_dir" ]] || { usage >&2; die "-o is required."; }
[[ -e "$input_path" ]] || die "Input does not exist: $input_path"
run_data_pipeline="$(normalize_bool "-D" "$run_data_pipeline")"
run_inference="$(normalize_bool "-P" "$run_inference")"
use_msa_server="$(normalize_bool "-M" "$use_msa_server")"
skip="$(normalize_bool "-S" "$skip")"
if [[ "$run_data_pipeline" == "false" && "$run_inference" == "false" ]]; then
    die "At least one of -D or -P must be true."
fi
case "$accelerator" in
    gpu|cpu|tpu) ;;
    *) die "-a must be gpu, cpu, or tpu, got: $accelerator" ;;
esac

if [[ -n "${BOLTZ_ENV_ACTIVATE:-}" ]]; then
    [[ -f "$BOLTZ_ENV_ACTIVATE" ]] || die "Activation script not found: $BOLTZ_ENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$BOLTZ_ENV_ACTIVATE"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
boltz_bin="${BOLTZ_BIN:-}"
if [[ -z "$boltz_bin" ]]; then
    for candidate in "$script_dir/.venv/bin/boltz" "$script_dir/venv/bin/boltz"; do
        if [[ -x "$candidate" ]]; then
            boltz_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$boltz_bin" ]]; then
    boltz_bin="$(command -v boltz || true)"
fi
[[ -n "$boltz_bin" && -x "$boltz_bin" ]] || die \
    "boltz executable not found; activate its environment or set BOLTZ_BIN."

cmd=(
    "$boltz_bin" predict "$input_path"
    --out_dir "$output_dir"
    -D "$run_data_pipeline"
    -P "$run_inference"
    --accelerator "$accelerator"
    --recycling_steps "$recycling_steps"
    --sampling_steps "$sampling_steps"
    --diffusion_samples "$diffusion_samples"
    --max_parallel_samples "$max_parallel_samples"
)

if [[ -n "$cache_dir" ]]; then
    cmd+=(--cache "$cache_dir")
fi
if [[ -n "$seeds" ]]; then
    cmd+=(--seeds "$seeds")
fi
if [[ "$use_msa_server" == "true" ]]; then
    cmd+=(--use_msa_server)
fi
if [[ "$skip" == "true" ]]; then
    cmd+=(--skip)
fi

if [[ "$run_inference" == "true" && "$accelerator" == "gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_devices"
    IFS=',' read -r -a visible_gpu_ids <<< "$gpu_devices"
    cmd+=(--devices "${#visible_gpu_ids[@]}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "Boltz command:"
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
