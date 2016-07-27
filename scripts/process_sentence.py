
def get_content(in_file, out_file):
    in_file = open(in_file,'r')
    out_file = open(out_file,'w')
    text = in_file.read()
    absracts = text.split('\r\n\r\n\r\n')
    buff = ''
    for abstract in absracts:
        abstract = abstract.strip('\r\n')
        content = abstract.split('\r\n\r\n')[4]
        content = content.replace('.',' ')
        content = content.replace(',',' ')
        content = content.replace('\r\n', ' ')
        content = content.replace('(',' ')
        content = content.replace(')',' ')
        content = content.replace('\\',' ')
        buff += content
        out_file.write(content)
    in_file.close()
    out_file.write(buff*2)
    out_file.close()
    return

def process_ncbi(in_path, class_out_path, word_out_path):
    in_file = open(in_path,'r')
    out_file = open(class_out_path,'w')
    contents = in_file.read()
    contents = contents.replace('.\t', ' ')
    contents = contents.split('\n')
    for content in contents:
        words =  content.split(' ')
        flag = False
        disease_word = []
        words[0] = words[0].split('\t')[1]
        for i in xrange(0,len(words)):
            if ('<category=\"SpecificDisease\">' in words[i]) or ('<category=\"Modifier\">' in words[i]) or ('<category=\"DiseaseClass\">' in words[i]) or ('<category=\"CompositeMention\">' in words[i]):
                words[i] = words[i].replace('<category=\"SpecificDisease\">','')
                words[i] = words[i].replace('<category=\"Modifier\">','')
                words[i] = words[i].replace('<category=\"DiseaseClass\">','')
                words[i] = words[i].replace('<category=\"CompositeMention\">','') 
                if '</category>' in words[i]:
                    out_file.write('B ' + words[i].replace('</category>','') + '\n')
                    continue
                flag = True
            if not flag:
                out_file.write('O ' + words[i] + '\n')
                continue
            else:
                disease_word.append(words[i])
            if '</category>' in words[i]:
                out_file.write('B ' + disease_word[0] + '\n')
                for j in xrange(1,len(disease_word)-1):
                    out_file.write('I ' + disease_word[j] + '\n')
                out_file.write('I ' + words[i].replace('</category>','') + '\n')
                disease_word = []
                flag = False
    word_out_file = open(word_out_path, 'w')
    for x in xrange(0,10):
        for content in contents:
            content = content.replace(',','')
            content = content.replace('.','')
            content = content.replace('(','')
            content = content.replace(')','')
            content = content.replace('[','')
            content = content.replace(']','')
            content = content.replace('<category=\"SpecificDisease\">','')
            content = content.replace('<category=\"Modifier\">','')
            content = content.replace('<category=\"DiseaseClass\">','')
            content = content.replace('<category=\"CompositeMention\">','')
            content = content.replace('</category>','')
            word_out_file.write(content.split('\t')[1])
    in_file.close()
    out_file.close()
    return

if __name__ == "__main__":
   # get_content('../data/abstract/finalresult.txt','../data/abstracts.txt')      
   process_ncbi('../data/NCBI/NCBI_corpus/NCBI_corpus_training.txt', '../data/NCBI/NCBI_corpus/class_train.txt','../data/NCBI/NCBI_corpus/word_train.txt' ) 


