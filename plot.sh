# Define the arrays of settings
depths=("18" "34" "50" "101" "152")
residuals=("True" "False")

# Loop over depth and residual settings and submit the job with sbatch
for depth in "${depths[@]}"; do
    for residual in "${residuals[@]}"; do
        cmd="python plot.py /home/hice1/yhao96/ResNet-country-guesser/config+history/resnet${depth}_residual=${residual}_transfer=False_history.json"
        echo "Executing: $cmd"
        eval $cmd
    done
done