# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from itertools import zip_longest
import editdistance as ed


def phoneme_distance(list1, list2):

    return ed.eval(list1, list2)
    # return sum(c1!=c2 for c1,c2 in zip_longest(list1,list2))

def sbl_distance(list1, list2):

    pos1 = []
    pos2 = []

    for pos, c in enumerate(list1):
        count = 0
        if c == '-':
            for k in list1[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos1.append(count)


    for pos, c in enumerate(list2):
        count = 0
        if c == '-':
            for k in list2[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos2.append(count)

    return len(list(set(pos1)^set(pos2)))


def stress_distance(list1, list2):

    pos1 = []
    pos2 = []
    pos3 = []
    pos4 = []

    for pos, c in enumerate(list1):
        count = 0
        if c == 'ˈ':
            for k in list1[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos1.append(count)


    for pos, c in enumerate(list1):
        count = 0
        if c == 'ˌ':
            for k in list1[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos2.append(count)


    for pos, c in enumerate(list2):
        count = 0
        if c == 'ˈ':
            for k in list2[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos3.append(count)


    for pos, c in enumerate(list2):
        count = 0
        if c == 'ˌ':
            for k in list2[:pos]:
                if k not in ('ˈ' , '-', 'ˌ'):
                    count += 1
            pos4.append(count)


    primary_mismatch = set(pos1)^set(pos3)
    secondary_mismatch = set(pos2)^set(pos4)

    return len(primary_mismatch & secondary_mismatch) + len(primary_mismatch ^ secondary_mismatch)

def cal_error(actual, target):
    actual = actual.split()
    target = target.split()
    stress_marks = ['ˌ','ˈ']
    syllable_mark = ['-']
    actual_phoneme = [c for c in actual if c not in stress_marks and c not in syllable_mark ]
    test_phoneme = [c for c in target if c not in stress_marks and c not in syllable_mark ]

    actual_phoneme_with_stress = [c for c in actual if c not in syllable_mark ]
    target_phoneme_with_stress = [c for c in target if c not in syllable_mark ]

    actual_phoneme_with_sbl = [c for c in actual if c not in stress_marks ]
    target_phoneme_with_sbl = [c for c in target if c not in stress_marks ]

    actual_primary_stress = [pos for pos, char in enumerate(actual_phoneme_with_stress) if char in stress_marks[1]]
    target_primary_stress = [pos for pos, char in enumerate(target_phoneme_with_stress) if char in stress_marks[1]]

    actual_secondary_stress = [pos for pos, char in enumerate(actual_phoneme_with_stress) if char in stress_marks[0]]
    target_secondary_stress = [pos for pos, char in enumerate(target_phoneme_with_stress) if char in stress_marks[0]]

    actual_syllable = [pos for pos, char in enumerate(actual_phoneme_with_sbl) if char in syllable_mark]
    target_syllable = [pos for pos, char in enumerate(target_phoneme_with_sbl) if char in syllable_mark]



    phoneme_error = phoneme_distance(actual_phoneme, test_phoneme)
    sbl_error = sbl_distance(actual, target)
    str_error = stress_distance(actual, target)
    
    # str_error = ed.eval(actual_primary_stress, target_primary_stress) + ed.eval(actual_secondary_stress, target_secondary_stress)

    # phoneme_length = max(len(actual), len(target))
    phoneme_length = len(actual)
    stress_len = len(actual_primary_stress) + len(actual_secondary_stress)
    syllable_len = len(actual_syllable)

    # print(actual)
    # print(target)
    # print(actual_phoneme)
    # print(test_phoneme)
    # print(actual_syllable)
    # print(target_syllable)
    # print(actual_primary_stress, actual_secondary_stress)
    # print(target_primary_stress, target_secondary_stress)
    # print(phoneme_error, phoneme_length)
    # print(sbl_error, syllable_len)
    # print(str_error, stress_len)

    # input(">")


    return (phoneme_error, sbl_error, str_error, phoneme_length, stress_len, syllable_len)



def main():
    fr0 = open("test_source.text","r", encoding='utf-8') #opens the input file
    fr1 = open("actual_target.text","r", encoding='utf-8') #opens the input file
    fr2 = open("test_target.text","r", encoding='utf-8') #opens the input file
    fw = open("test_target.csv","w", encoding='utf-8') #opens the output file
    fw1 = open("result.text", "w", encoding='utf-8')

    sources = fr0.readlines()
    actual_target = fr1.readlines()
    test_target = fr2.readlines()

    result_tuples = zip(sources, actual_target, test_target)

    WER = 0.0
    PER = 0.0
    StER = 0.0
    SyER = 0.0

    total_words = 0
    total_phones = 0
    total_syllable = 0
    total_stress_marks = 0

    with fw:  
        writer = csv.writer(fw)
        header = ['source Words', 'Ground Truth', 'G2P Output', 'Word_Error', 'Phoneme_Error', 'Syllable_Error', 'Stress Error']
        writer.writerow(header)

        for source, actual, output in result_tuples:
            row = []
            word_error, syllable_error, stress_error, phoneme_length, stress_len, syllable_len = cal_error(actual, output)

            # print(word_error, syllable_error, stress_error, phoneme_length, stress_len, syllable_len)
            # input(">")

            if word_error !=0:
                WER +=1
                PER += word_error
                we = 1
            else:
                we = 0

            if syllable_error != 0:
                SyER += 1

            if stress_error != 0:
                StER += 1

            total_words += 1
            total_phones += phoneme_length
            total_syllable += syllable_len
            total_stress_marks += stress_len

            row.append(source)
            row.append(actual)
            row.append(output)
            row.append(str(we))
            row.append(str(word_error))
            row.append(str(syllable_error))
            row.append(str(stress_error))

            writer.writerow(row)

            total_WER = (WER/total_words)
            total_PER = (PER/total_phones)
            total_SyER = (SyER/total_words)
            total_StER = (StER/total_words)


    performance = "Word_Error_Rate: {0: f} \nPhoneme_Error_Rate: {1: f} \nSyllable_Error_Rate: {2: f} \nStress_Error_Rate: {3: f}".format(total_WER, total_PER, total_SyER, total_StER)

    fw1.write(performance)

    fr0.close()
    fr1.close()
    fr2.close()
    fw.close()
    fw1.close()

if __name__ == '__main__':
    main()