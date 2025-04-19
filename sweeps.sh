# Define the arrays of settings
depths=("18" "34" "50" "101" "152")
residuals=("true" "false")
dataset=""
batch_size="64"
epochs="100"

# Define long options for getopt
OPTIONS=$(getopt -o "" -l "dataset:,batch_size:,epochs:" -- "$@")

# Check if getopt exited with nonzero exit code
if [ $? -ne 0 ]; then
    echo "Usage: $0 --dataset <country|geo> --batch_size <integer size, default 64> --epochs <integer, default 100>"
    exit 1
fi

eval set -- "$OPTIONS"

# Extract CLI arguments
while true; do
    case "$1" in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
	--epochs)
	    epochs="$2"
	    shift 2
	    ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown argument $1"
            exit 1
            ;;
    esac
done

# Verify that the provided argument is a valid dataset name
if [ -z "$dataset" ]; then
    echo "Missing required argument \"--dataset\": select either \"country\" or \"geo\" as dataset"
    exit 1
elif [ "$dataset" != "country" -a "$dataset" != "geo" ]; then
    echo "Argument \"--dataset\" does not match either \"country\" or \"geo\""
    exit 1
fi

# Verify that batch size is an integer
if [[ ! "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Batch size \"$batch_size\" is not a valid integer"
    exit 1
fi

# Verify that epochs is an integer
if [[ ! "$epochs" =~ ^[0-9]+$ ]]; then
    echo "Number of epochs \"$epochs\" is not a valid integer"
    exit 1
fi

# Loop over depth and residual settings and submit the job with sbatch
for depth in "${depths[@]}"; do
    for residual in "${residuals[@]}"; do
        cmd="sbatch run.sh python main.py --depth $depth --residual $residual --dataset $dataset --batch-size $batch_size --epochs $epochs"
        echo "Executing: $cmd"
        eval $cmd
    done
done
