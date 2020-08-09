import random


def generate_cross_validation_dataset(inpath, domain, k_fold):
    datalist = []
    with open(inpath, 'r') as fin:
        idx = 0
        dataunit = []
        for line in fin:
            if idx % 6 == 5:
                dataunit.append(line)
                datalist.append(dataunit)
                dataunit = []
            else:
                dataunit.append(line)
            idx += 1
    random.shuffle(datalist)
    batch_num = len(datalist) / k_fold
    trainlist, devlist = [], []
    for i in range(k_fold):
        start = i * batch_num
        end = (i+1) * batch_num
        if i == k_fold-1: end = len(datalist)
        devlist.append(datalist[start:end])
        trainlist.append(datalist[:start]+datalist[end:])

    def sentiment_statistics(name, dataset):
        pos, neu, neg = 0, 0, 0
        for senlist in dataset:
            label = int(senlist[1].strip())
            if label == 1: pos += 1
            elif label == 0: neu += 1
            elif label == -1: neg += 1
        print(name+'\t\tpos:{}\t neu:{}\t neg:{}'.format(pos, neu, neg))

    def writetofile(dataset, datapath):
        with open(datapath, 'a') as fout:
            for senlist in dataset:
                for unit in senlist:
                    for _ in unit:
                        fout.write(_)

    for i in range(k_fold):
        print('\n{}-fold statistics:'.format(i))
        sentiment_statistics('training', trainlist[i])
        sentiment_statistics('dev', devlist[i])

    order = input('\ngenerate or not? y/n\n')
    if order == 'y':
        for i in range(k_fold):
            trainpath = 'data/' + domain + '/' + domain + '_train_'+str(i+1)+'.txt'
            devpath = 'data/' + domain + '/' + domain + '_dev_'+str(i+1)+'.txt'
            writetofile(trainlist[i], trainpath)
            writetofile(devlist[i], devpath)


if __name__ == "__main__":
    # generate_cross_validation_dataset('data/restaurant/rest_2014_multitop_train_new_1.txt', 'restaurant', 5)
    # generate_cross_validation_dataset('data/laptop/laptop_2014_multitop_train.txt', 'laptop', 5)
    generate_cross_validation_dataset('data/twitter/twitter_2014_multitop_train_new_1.txt', 'twitter', 5)
