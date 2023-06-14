# Materials-Synthesis-PU-GNN
Personal Re-construction of "Synthesizability-PU-CGCNN".

This is personal re-construction of 
[Structure-Based Synthesizability Prediction of Crystals Using Partially Supervised Learning, *J. Am. Chem. Soc. 2020, 142, 18836−18843*](https://pubs.acs.org/doi/pdf/10.1021/jacs.0c07384).
The code of the original can be found at [here](https://github.com/kaist-amsg/Synthesizability-PU-CGCNN).  
The most part of code are the same as the original, so I recommend you to see their code first, if you are struggling in using this works.

### Difference from the original
Although this repo is an *re-Implementation* work, there exist some **differents** from the original. The most of them comes from:
* Training Framework(PU learning)
* CGCNN Implementation

#### - Training Framework(PU Learning)
In the original work, the authors used PU-bagging in **parallel** manner. It means, they trained 50(by default) independent model within each PU-bagging split. However, I used PU-bagging in **Seria*l* manner. I trained 1 model, with iterative splitting PU-bag. In other words, the result of the original is *the ensemble of 50 models each with 30 epochs(by default)*. And, the result of my work is *the one model with 1500(50\*30) epochs*.

At the time of my implementation, it was just a mis-understanding of the [paper](https://pubs.acs.org/doi/pdf/10.1021/jacs.0c07384). By the way, the result of that work performs on far as the originals(you can see its result in [here](https://github.com/SJ1115/Materials-Synthesis-PU-GNN/blob/main/result/score/old.png)). Sometimes it can be more attractive since we only have to keep 1 model spaces.
If you want to use this training framework, you can train it by:
```
trainer = Trainer(model, criterion, optimizer, ...)
trainer.PU_train("id_prop_train.csv", result_file="result.pt")
```
Then you can find your model in `result.pt`.

After then, I re-implemented the PU-learning as like the original(*but I have not tested it*). If you want to use this training framework, you can train it by:
```
trainer = Trainer(model, criterion, optimizer, ...)
trainer.PU_ensemble_train("id_prop_train.csv", result_dir="/result")
```
Then you can find your ensemble-models as `/result/1.pt`, `/result/2.pt`, `/result/3.pt`, ...

#### - CGCNN Implementation

The original work uses CGCNN architecture, which is exactly the same from its own [repository](https://github.com/txie-93/cgcnn).
In my implementation, the architecture is still the same, but their are some minor differences:
- For simplicity, I used `config` argument to monitor all argument at once. You can find it in `config.py`.
- In original repo, `id_prop.csv` and `atom_init.json` should locate in `.cif` file folder.
It makes hard to check or control `.csv` and `.json` file, especially for the users with GUI, such as Jupyter(like me), since openning folder with large amount of files can yields some delay in loading filenames.
In here, those 2 file locate in `/data` folder, not in `/data/cif/`. I think it can help you slightly better.
