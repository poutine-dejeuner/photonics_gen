# photo-gen
Implementation and comparison of different diffusion models for image generation in low data setting with objective high FOM and generation diversity.

![image](https://github.com/poutine-dejeuner/photonics-gen/diffusion.png)

Download data with

`wget --no-check-certificate 'https://drive.google.com/file/d/1-ZTBtsQsQ6Sk2VR-zNRhTDjlmzc6bsd2/view?usp=sharing' -O images.zip`

`unzip images.zip`
`path = ~/scratch/nanophoto/topoptim/fulloptim/`
`mkdir -R path`
`mv images.npy path` 

To train models, use train.py or to run on slurm use `sbatch setoff.sh main.py`.

main.py uses hydra. Default configs can be overridden by 
    `python train.py model=wgan`

To compare several trained models, use photo_gen/evaluation/compare_models.py
with argument evaluation.datasets=datasets.yaml and datasets.yaml a config file
containing for each dataset
```
datasets:
  - name: my_dataset
  - path: path/to/dataset/images.npy
  - n_samples: number_of_training_samples
```

The script will save plots in `output/<time>/metric_plots`.

