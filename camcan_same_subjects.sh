for (( i=1; i<=20; i++ ))
do
    sbatch -c 64 -p normal,parietal --wrap "python camcan_same_subjects_bootstrap.py -s $i"
done
