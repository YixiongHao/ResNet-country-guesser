# Define the arrays of settings
depths=("18" "34" "50" "101" "152")
residuals=("True" "False")

# Check if dir is provided
if [ -z "$1" ]; then
    echo "Missing history directory argument"
    echo "Usage: $0 <directory>"
    exit 1
fi

dir=$(realpath "$1")

if [ ! -d "$dir" ]; then
    echo "Invalid directory: $dir"
    exit 1
fi

# Loop over depth and residual settings and submit the job with sbatch
for depth in "${depths[@]}"; do
    for residual in "${residuals[@]}"; do
        cmd="python plot.py $dir/resnet${depth}_residual=${residual}_transfer=False_history.json"
        echo "Executing: $cmd"
        eval $cmd
    done
done
