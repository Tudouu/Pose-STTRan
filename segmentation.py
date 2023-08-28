#from stanfordcorenlp import StanfordCoreNLP
import re

str='aefarvb asrdgvawr'
res=re.findall('far',str)
print(res)
exit()
#nlp = StanfordCoreNLP('./stanford-corenlp-4.2.0')
nlp=StanfordCoreNLP(r'/media/jocker/disk2/AG/stanford-corenlp-4.3.2')
#sentence = "Steers turns in a snappy screenplay that curls at the edges ; it 's so clever you want to hate it ."
#sentence='A woman takes a picture of another woman wearing a flower lei.'
#         1   2     3   4    5    6     7      8     9     10   11    12
#[('ROOT', 0, 3), ('det冠词', 2, 1 a woman), ('nsubj名词主语', 3, 2 woman takes), ('det冠词', 5, 4 a picture), ('obj宾语', 3, 5 takes picture),
# ('case', 8, 6), ('det冠词', 8, 7 another woman), ('nmod复合名词修饰', 5, 8 正在拍照(picture)的woman), ('acl', 8, 9), ('det', 11, 10), ('obj', 9, 11), ('advmod', 9, 12), ('punct', 3, 13)]
#sentence='A woman serves food to a man holding a plate.'
#         1   2     3      4   5  6  7      8   9   10
#[('ROOT', 0, 3), ('det', 2, 1), ('nsubj', 3, 2), ('obj', 3, 4), ('case', 7, 5), ('det', 7, 6), ('obl', 3, 7), ('acl', 7, 8), ('det', 10, 9), ('obj', 8, 10), ('punct', 3, 11)]
#--------------------------------------------------------acl---------------------------------------------
sentence='A white man looks at a woman who is riding a bicycle.'
#         1  2     3    4   5  6  7    8   9  10    11   12
ss=sentence.split(' ')
priority=[]
result = nlp.dependency_parse(sentence)
for i in range(len(result)):
    s=str(result[i][0])
    if re.findall('acl',s):
        print(s)
        print(result[i][1],result[i][2])
        print(ss[result[i][2]-1])
        priority.append(result[i][2]-1)
priority.append(result[0][2]-1)
print(priority)
print(result)
#[('ROOT', 0, 3), ('det', 2, 1), ('nsubj', 3, 2), ('case', 6, 4), ('det', 6, 5), ('obl', 3, 6), ('nsubj', 9, 7), ('aux', 9, 8), ('acl:relcl', 6, 9), ('det', 11, 10), ('obj', 9, 11), ('punct', 3, 12)]
#data = pd.read_csv('data/train_1.csv')

