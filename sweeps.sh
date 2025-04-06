# Define the arrays of settings
depths=("18" "34" "50" "101" "152")
residuals=("true" "false")
dataset="$1"

# Verify that the provided argument is a valid dataset name
if [ -z "$dataset" ]; then
    echo "Missing argument: select either \"country\" or \"geo\" as dataset"
    exit 1
elif [ "$dataset" != "country" -a "$dataset" != "geo" ]; then
    echo "Argument does not match either \"country\" or \"geo\" for dataset"
    exit 1
fi

# Loop over depth and residual settings and submit the job with sbatch
for depth in "${depths[@]}"; do
    for residual in "${residuals[@]}"; do
        cmd="sbatch run.sh python main.py --depth $depth --residual $residual --dataset $dataset"
        echo "Executing: $cmd"
        eval $cmd
    done
done
