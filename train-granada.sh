eval "$(conda shell.bash hook)"
conda activate /home/jlsalazar/envs/self-distill


python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/granada/01'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/granada/02'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/granada/03'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/granada/04'

python train.py --phase warmup \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/warmup/granada/05'

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/granada/01' \
                --warmup_output 'experiments/warmup/granada/01' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/granada/02' \
                --warmup_output 'experiments/warmup/granada/02' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/granada/03' \
                --warmup_output 'experiments/warmup/granada/03' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/granada/04' \
                --warmup_output 'experiments/warmup/granada/04' \
                --epc 5

python train.py --phase iterative \
                --supervised_dir='/home/jossalgon/datasets/weapons/granada-3000' \
                --unsupervised_dir='/home/jossalgon/datasets/unsupervised/instagram-resized800' \
                --output 'experiments/iterative/granada/05' \
                --warmup_output 'experiments/warmup/granada/05' \
                --epc 5

