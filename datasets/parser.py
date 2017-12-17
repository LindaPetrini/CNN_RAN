
def parse(input,output):
    # sentences=[]
    # targets=[]
    lines=[]
    with open('./'+input, 'r') as f:
        for l in f:
            line = l.split(",") #NOTE in 2013 there is one more ID at beginning of each tweet
            # targets.append(target)
            # sentences.append(sentence)
            target = line[1]
            sentence = line[3:]
            tweet = (",".join(sentence)).strip()+"\n"
            lines.append(target+"\t"+tweet)
            #print(lines[-1])
    
    with open('./'+output,'w') as f:
        for line in lines:
            f.write(line)
       

parse('Sentiment Analysis Dataset.csv','emb_dataset.txt')