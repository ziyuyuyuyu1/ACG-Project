# ACG Project

The result page is <a href="https://ziyuyuyuyu1.github.io/ACG-UI/">here</a>.

The code for geometry generation is in **./lib** directory.

The code for texture generation is in **./nvdiffrec** directory.

The checkpoints are <a href="https://huggingface.co/ziyuyuyuyu1/ACG-class-cond-ckpt/">here</a>.


### Mesh Generation
<!-- Code --> 

First execute:

``` bash
python3 main_diffusion.py --config=./configs/res64_classifier.py  --mode=uncond_gen --config.eval.eval_dir=$OUT_DIR --config.eval.ckpt_path=$PATH_TO_CHECKPOINT --config.eval.classifier_path=$PATH_TO_CLASSIFIER
```

Then you can get a ```0.npy``` file in ```$OUT_DIR```, which should be of size ```(batch_size, 4, 64, 64, 64)```. The default batch size here is 5, and 5 objects of different types (airplane, chair, table, rifle, car) would be generated (In the file ```./lib/diffusion/evaler.py``` the 5 types corresponds to the context ```[[1],[2],[3],[4],[5]]```). Then you should run

``` bash
cd nvdiffrec
python3 eval.py --config configs/res64.json --sample-path $PATH_TO_NPY --out-dir $OUTPUT_DIR
```

After that you can get the resulting ```.obj``` files.

### Texture Generation