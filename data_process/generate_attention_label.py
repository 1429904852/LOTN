def read1(input_file1):
    lines = open(input_file1).readlines()
    opinion = []
    total = []
    for i in range(len(lines)):
        sen = lines[i].split()
        sen_len = len(sen)
        average = 1.0 / sen_len
        for j in range(len(sen)-1):
            if float(sen[j]) >= average:
                opinion.append(1)
            else:
                opinion.append(0)
        opinion.append(0)
        total.append(opinion)
        opinion = []
    return total


def write(input_file1, output_file_2):
    opinion = read1(input_file1)
    fp = open(output_file_2, 'w')
    for seg in range(len(opinion)):
        for seg1 in range(len(opinion[seg])):
            if seg1 == len(opinion[seg])-1:
                fp.write(str(opinion[seg][seg1]))
            else:
                fp.write(str(opinion[seg][seg1]) + " ")
        fp.write("\n")


if __name__ == "__main__":

    file1 = 'data/14res/14res_train_att.txt'
    file2 = 'data/14res/att_train.txt'
    write(file1, file2)

    file1 = 'data/14res/14res_test_att.txt'
    file2 = 'data/14res/att_test.txt'
    write(file1, file2)

    file1 = 'data/14lap/14lap_train_att.txt'
    file2 = 'data/14lap/att_train.txt'
    write(file1, file2)

    file1 = 'data/14lap/14lap_test_att.txt'
    file2 = 'data/14lap/att_test.txt'
    write(file1, file2)

    file1 = 'data/15res/15res_train_att.txt'
    file2 = 'data/15res/att_train.txt'
    write(file1, file2)

    file1 = 'data/15res/15res_test_att.txt'
    file2 = 'data/15res/att_test.txt'
    write(file1, file2)

    file1 = 'data/16res/16res_train_att.txt'
    file2 = 'data/16res/att_train.txt'
    write(file1, file2)

    file1 = 'data/16res/16res_test_att.txt'
    file2 = 'data/16res/att_test.txt'
    write(file1, file2)