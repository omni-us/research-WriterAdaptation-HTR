# Unsupervised Writer Adaptation for Synthetic-to-Real Handwritten Word Recognition

An unsupervised writer adaptation approach that is able to automatically adjust a generic handwritten word recognizer, fully trained with synthetic fonts, towards a new incoming writer.

![Architecture](https://user-images.githubusercontent.com/9562709/78949930-7ea75000-7acd-11ea-9e11-d081fbcd50a5.png)

[Unsupervised Writer Adaptation for Synthetic-to-Real Handwritten Word Recognition](http://openaccess.thecvf.com/content_WACV_2020/papers/Kang_Unsupervised_Writer_Adaptation_for_Synthetic-to-Real_Handwritten_Word_Recognition_WACV_2020_paper.pdf)<br>
Lei Kang, Marçal Rusiñol, Alicia Fornés, Pau Riba, and Mauricio Villegas<br>
In The IEEE Winter Conference on Applications of Computer Vision, pp. 3502-3511. 2020.

## Software environment:

- Ubuntu 16.04 x64
- Python 3
- PyTorch 0.3

## Dataset preparation

We are using [60K synthetic word images](https://github.com/kikones34/handwritten-document-synthesizer) as source data and popular real handwritten word datasets ([GW](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database), [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database), [RIMES](http://www.a2ialab.com/doku.php?id=rimes_database:start), [Esposalles](http://dag.cvc.uab.es/the-esposalles-database/) and
[CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/)) as target data. 

## Note for the project:

We have shown the results of different experimental settings in the paper. However, in order to make the code clean and easy to understand, the code we show in this repo is the domain adaptation from synthetic word images to IAM word dataset. For the other experiments, it would be easy to reproduce with some minor changes.

## Before training:

We use `tasas_cer.sh` and `tasas_wer.sh` to calculate the character error rate and word error rate, which have dependency on an [external tool](https://github.com/omni-us/research-seq2seq-HTR/tree/master/utils). So make sure that this tool has been properly installed and also double check the url to be your correct location inside the two shell scripts.

## How to train it?

Once both of the synthetic data and real data are prepared, you need to denote the correct urls in the file `loadData6_vgg.py`, then you are ready to go by running:

```bash
./run_train.sh
```
**Note**: Which GPU to use or which epoch you want to start from could be set in this shell script. (Epoch ID corresponds to the weights that you want to load in the folder `save_weights`)

## How to test it?

```bash
./run_test.sh
```

## Citation

If you use the code for your research, please cite our paper:

```
@inproceedings{kang2020unsupervised,
  title={Unsupervised Writer Adaptation for Synthetic-to-Real Handwritten Word Recognition},
    author={Kang, Lei and Rusi{\~n}ol, Mar{\c{c}}al and Forn{\'e}s, Alicia and Riba, Pau and Villegas, Mauricio},
      booktitle={The IEEE Winter Conference on Applications of Computer Vision},
        pages={3502--3511},
          year={2020}
}
```
