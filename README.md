# ChatBot
Chinese/English ChatBot. 
## Corpus
```bash
$ mkdir dataset
```
### Cornell Movie-Dialogs Corpus
This corpus contains a metadata-rich collection of fictional conversations extracted from raw movie scripts:

- 220,579 conversational exchanges between 10,292 pairs of movie characters
- involves 9,035 characters from 617 movies
- in total 304,713 utterances

```bash
$ wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
$ unzip -n cornell_movie_dialogs_corpus.zip -d dataset 
$ rm cornell_movie_dialogs_corpus.zip
```  

### Xiaohuangji Corpus
```bash
$ wget https://github.com/fate233/dgk_lost_conv/raw/master/results/xiaohuangji50w_nofenci.conv.zip
$ unzip -n xiaohuangji50w_nofenci.conv.zip -d dataset 
$ rm xiaohuangji50w_nofenci.conv.zip
```
## Get Start
### Generate word dict and tensor-format dataset
```bash
$ python initen.py -o data/en
$ python initchs.py -o data/chs
``` 
### Train
```bash
$ python train.py -i data/en -o output/en
$ python train.py -i data/chs -o output/chs
```
### Evaluation
```bash
$ python eval.py --en
$ python eval.py 
```