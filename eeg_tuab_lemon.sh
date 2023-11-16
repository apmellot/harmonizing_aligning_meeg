for (( i=1; i<=20; i++ ))
do
    sbatch -c 64 -p normal,parietal --wrap "python eeg_tuab_lemon.py -s $i"
done
