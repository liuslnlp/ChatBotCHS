# ChatBot
Chinese/English ChatBot.  
**English mode**
```
> how are you?
Bot: i m fine . . . .
> where are you from?
Bot: i m not . . . . .
> do you love me?
Bot: i love you . . .
> play with me
Bot: i m sorry . . . . .
> bye bye
Bot: bye bye bye .
```
**Chinese mode**
```
> 你好吗？
Bot: 你好，我好，大家好
> 你有病吗？
Bot: 你有药啊？
> 好想吃饭
Bot: 我也想吃肿么办。我
> 你在哪上学？
Bot: 我在你家
```
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
This corpus contains 500,000 chinese conversations.
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
$ python eval.py -v data/en -c output/en --en
$ python eval.py -v data/chs -c output/chs
```