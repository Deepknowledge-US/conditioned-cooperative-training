eval "$(conda shell.bash hook)"
conda activate /home/jlsalazar/envs/self-distill

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/youtube/01'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/youtube/02'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/youtube/03'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/youtube/04'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/youtube/05' 

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/youtube/01' \
                --warmup_output 'experiments/warmup/youtube/01' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/youtube/02' \
                --warmup_output 'experiments/warmup/youtube/02' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/youtube/03' \
                --warmup_output 'experiments/warmup/youtube/03' \
                --epc 5

python train.py --device 'cuda:1' --phase iterative \
                --supervised_dir='/data/jlsalazar/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/data/jlsalazar/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/youtube/04' \
                --warmup_output 'experiments/warmup/youtube/04' \
                --epc 5

python train.py --device 'cuda:1' --phase iterative \
                --supervised_dir='/data/jlsalazar/datasets/weapons/YouTube-GDD' \
                --unsupervised_dir='/data/jlsalazar/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/youtube/05' \
                --warmup_output 'experiments/warmup/youtube/05' \
                --epc 5
