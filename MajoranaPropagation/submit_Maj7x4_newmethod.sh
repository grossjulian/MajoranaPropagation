#!/bin/bash

# Parameters
run_name="New_method"
nx=7
ny=4
t=1.0
dt=0.1
save_dir_base="/cluster/scratch/jugross/MajoranaData/${nx}x${ny}/${run_name}" #"./saves/FH"

mem_per_cpu=6 # in GB
from_checkpoint=false

# Create saves and logs directories
log_dir="${save_dir_base}/logs"
save_dir="${save_dir_base}/Data/"
run_stamp=$(date +%Y%m%d_%H%M%S)
jobscript_dir="${log_dir}/submitted_scripts_${run_stamp}"
jobids_file="${log_dir}/jobids.json"
mkdir -p "$log_dir"
mkdir -p "$save_dir"
mkdir -p "$jobscript_dir"

# Initialize jobids.json if it doesn't exist
if [ ! -f "$jobids_file" ]; then
    cat > "$jobids_file" << 'EOJSON'
{
  "job_submissions": []
}
EOJSON
fi

append_job_submission() {
    local job_number="$1"
    local job_id="$2"
    local n_layers="$3"
    local u_value="$4"
    local min_abs_coeff_value="$5"
    local max_penalties_value="$6"
    local save_path_value="$7"
    local submit_timestamp_value="$8"
    local submit_run_stamp_value="$9"

    if ! command -v jq &> /dev/null; then
        return 1
    fi

    local has_record
    has_record=$(jq -r --argjson job_number "$job_number" \
        'any(.job_submissions[]?; .job_number == $job_number)' "$jobids_file")

    if [ "$has_record" = "true" ]; then
        temp_json=$(jq \
            --argjson job_number "$job_number" \
            --arg job_id "$job_id" \
            --arg n_layers "$n_layers" \
            --arg u "$u_value" \
            --arg min_abs_coeff "$min_abs_coeff_value" \
            --arg max_penalties "$max_penalties_value" \
            --arg save_path "$save_path_value" \
            --arg submit_timestamp "$submit_timestamp_value" \
            --arg submit_run_stamp "$submit_run_stamp_value" \
            '.job_submissions |= map(
                if .job_number == $job_number then
                    .job_id = $job_id
                    | .job_ids = ((.job_ids // []) + [$job_id])
                    | .job_attempts = ((.job_attempts // []) + [{
                        "job_id": $job_id,
                        "submit_timestamp": $submit_timestamp,
                        "submit_run_stamp": $submit_run_stamp
                      }])
                                        | .U = ($u | tonumber)
                    | .min_abs_coeff = $min_abs_coeff
                    | .max_penalties = $max_penalties
                    | .n_layers = $n_layers
                    | .save_path = $save_path
                    | .submit_timestamp = $submit_timestamp
                    | .submit_run_stamp = $submit_run_stamp
                else
                    .
                end
            )' "$jobids_file")
    else
        temp_json=$(jq \
            --argjson job_number "$job_number" \
            --arg job_id "$job_id" \
            --arg n_layers "$n_layers" \
            --arg u "$u_value" \
            --arg min_abs_coeff "$min_abs_coeff_value" \
            --arg max_penalties "$max_penalties_value" \
            --arg save_path "$save_path_value" \
            --arg submit_timestamp "$submit_timestamp_value" \
            --arg submit_run_stamp "$submit_run_stamp_value" \
            '.job_submissions += [{
                "job_number": $job_number,
                "job_id": $job_id,
                "job_ids": [$job_id],
                "job_attempts": [{
                  "job_id": $job_id,
                  "submit_timestamp": $submit_timestamp,
                  "submit_run_stamp": $submit_run_stamp
                }],
                                "U": ($u | tonumber),
                "min_abs_coeff": $min_abs_coeff,
                "max_penalties": $max_penalties,
                "n_layers": $n_layers,
                "save_path": $save_path,
                "submit_timestamp": $submit_timestamp,
                "submit_run_stamp": $submit_run_stamp
              }]' "$jobids_file")
    fi

    echo "$temp_json" > "$jobids_file"
}

# Generate parameter combinations and submit jobs
i=0

for n_layers in 20; do
for U in 4; do

    if [ "$U" = "4" ]; then
        julia_threads=20
        time_limit="200:00:00" #200:00:00
        mac_exponents=(-6)
        max_penalties_values=(3 4 5 6) #4 6 8
        #partition="bigmem" 
        mem_per_cpu_mac=45 #900/20
    else
        mac_exponents=(-4 -6 -8 -10)
        max_penalties_values=("")   # sentinel: no max_penalties for U=0
        julia_threads=5
        time_limit="72:00:00"
        #partition="normal.12"
        mem_per_cpu_mac=6 #150/5
    fi

    for exp in "${mac_exponents[@]}"; do
        min_abs_coeff=$(awk -v e="$exp" 'BEGIN { printf "%.0e", 10^e }')

    for max_penalties in "${max_penalties_values[@]}"; do
        i=$((i + 1))

        jobscript="${jobscript_dir}/job_${i}.sh"

        # Build optional --max_penalties flag
        if [ -n "$max_penalties" ]; then
            max_penalties_flag="--max_penalties $max_penalties"
        else
            max_penalties_flag=""
        fi

        cat > "$jobscript" << EOF   
#!/bin/bash
#SBATCH --time=$time_limit
#SBATCH --cpus-per-task=$julia_threads
#SBATCH --mem-per-cpu=${mem_per_cpu_mac}G
#SBATCH --output=$log_dir/out_${i}.log
#SBATCH --error=$log_dir/err_${i}.log
#SBATCH --mail-type=FAIL

cd $PWD
julia -t $julia_threads Majorana7x4_newmethod.jl \\
    --nx $nx \\
    --ny $ny \\
    --U $U \\
    --t $t \\
    --n_layers $n_layers \\
    --dt $dt \\
    --min_abs_coeff $min_abs_coeff \\
    --from_checkpoint $from_checkpoint \\
    --save_dir $save_dir\\
    $max_penalties_flag
EOF

        echo "Submitting job $i:nx=$nx, ny=$ny, U=$U, t=$t, n_layers=$n_layers, dt=$dt, min_abs_coeff=$min_abs_coeff, max_penalties=$max_penalties, from_checkpoint $from_checkpoint, save_dir=$save_dir, mem=${mem_per_cpu_mac}G"
        job_id=$(sbatch --parsable "$jobscript")
        submit_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

            append_job_submission "$i" "$job_id" "$n_layers" "$U" "$min_abs_coeff" "$max_penalties" "$save_dir" "$submit_time" "$run_stamp"
    done  # max_penalties
    done  # min_abs_coeff
done  # U
done  # n_layers

echo "All $i jobs submitted!"
