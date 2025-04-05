# Define the arrays of settings
depths=("18" "34" "50" "101" "152")
residuals=("true" "false")

# Loop over depth and residual settings and submit the job with sbatch
for depth in "${depths[@]}"; do
    for residual in "${residuals[@]}"; do
        cmd="sbatch run.sh python main.py --depth $depth --residual $residual"
        echo "Executing: $cmd"
        eval $cmd
    done
done