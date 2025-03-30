# EQAFold Public Release

This repository contains all necessry code, environment definitions, and examples necessary for running EQAFold.

## Running Examples

### Data setup
Each target to be passed through the network must have a subdirectory in a master transformer directory. Within each target subdirectory (example here:XXXX) the following naming scheme most be present:
```
.
├── XXXX
│   ├── XXXX_esm_attn.npz
│   ├── XXXX_esm_layer.npz
│   ├── XXXX.fasta
│   ├── XXXX_renum.pdb
│   ├── XXXX_rmsf.npz
│   └── model_1.npz
├── YYYY
│   └── ....
└── ZZZZ
    └── ....
```
All targets that you would like to pass to the model must both:
1) Exist within this directory, with all necessary files present, and
2) Be listed in an input list (here, ```example.list```)

### Environment
The environment is defined in ```env.yaml```, and can be installed via conda as follows:
```
conda env creaate -n "eqafold" -f env.yaml --yes
conda activate eqafold
```
### Running
We have provided an input fine ```test.sh``` that will run EQAFold on three example sequences listed in the ```example``` directory. By running the script, you will generate the three structures in the ```example_output``` subsirectory.


```
conda activate eqafold
bash test.sh
ls example/example_output/
2AWFA_pred_all.pdb  7Y3HA_pred_all.pdb  8A55A_pred_all.pdb  opt.txt
```


EQAFold was trained on a single 48 GB A100, with at least 32 GB of RAM and 10 available cores.




## Dataset Generation Instructions
EQAFold is reliant on features from programs such as ESM2, that we can't distribute ourselves due to licencing. However, we describe the process of running and generating your dataset entries below.


```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
batch_converter = alphabet.get_batch_converter()

def esm2_t33(seq):
    data = [("protein1", ''.join(seq))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Run model
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=list(range(1,33+1)), return_contacts=True)

    #Extract Attentions
    attentions = results["attentions"] #(1, 33, 20, L+2, L+2)
    attentions = attentions.cpu().numpy()

    layerwise_representations = (
    torch.stack([
    results["representations"][i][0,1:len(seq)+1].mean(-1) for i in results["representations"]
    ]).transpose(0,1).cpu().numpy()
    ) # L X 33
    return attentions, layerwise_representations
```


