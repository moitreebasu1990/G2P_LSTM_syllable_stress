# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import editdistance as ed




def cal_error(actual, target):
    actual = actual.replace(" ", "")
    target = target.replace(" ", "")
    stress_marks = ['ˌ','ˈ']
    syllable_mark = ['-']
    actual_phoneme = "".join(c for c in actual if c not in stress_marks and c not in syllable_mark )
    test_phoneme = "".join(c for c in target if c not in stress_marks and c not in syllable_mark )

    actual_phoneme_with_stress = "".join(c for c in actual if c not in syllable_mark )
    target_phoneme_with_stress = "".join(c for c in target if c not in syllable_mark )

    actual_phoneme_with_sbl = "".join(c for c in actual if c not in stress_marks )
    target_phoneme_with_sbl = "".join(c for c in target if c not in stress_marks )

    actual_stress = [pos for pos, char in enumerate(actual_phoneme_with_stress) if char in stress_marks]
    target_stress = [pos for pos, char in enumerate(target_phoneme_with_stress) if char in stress_marks]

    actual_syllable = [pos for pos, char in enumerate(actual_phoneme_with_sbl) if char in syllable_mark]
    target_syllable = [pos for pos, char in enumerate(target_phoneme_with_sbl) if char in syllable_mark]

    # print(actual)
    # print(target)
    # print(actual_phoneme)
    # print(test_phoneme)
    # print(actual_syllable)
    # print(target_syllable)
    # print(actual_stress)
    # print(target_stress)

    # input(">")

    phoneme_error = ed.eval(actual_phoneme, test_phoneme)
    sbl_error = ed.eval(actual_syllable, target_syllable)
    str_error = ed.eval(actual_stress, target_stress)

    phoneme_length = len(actual)
    stress_len = len(actual_stress)
    syllable_len = len(actual_syllable)

    return (phoneme_error,sbl_error ,str_error ,phoneme_length ,stress_len ,syllable_len )



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
    SER = 0.0
    SyER = 0.0

    total_words = 0
    total_phones = 0
    total_syllable = 0
    total_stress_marks = 0

    with fw:  
        writer = csv.writer(fw)
        header = ['source Words', 'System Output', 'Actual Output', 'Word_Error', 'Phoneme_Error', 'Syllable_Error', 'Stress Error']
        writer.writerow(header)

        for source, actual, output in result_tuples:
            row = []
            word_error, syllable_error, stress_error, phoneme_length, stress_len, syllable_len = cal_error(actual, output)

            if word_error !=0:
                WER +=1
                PER += word_error
                we = 1
            else:
                we = 0

            if syllable_error != 0:
                SyER += syllable_error

            if stress_error != 0:
                SER += stress_error

            total_words += 1
            total_phones += phoneme_length
            total_syllable += syllable_len
            total_stress_marks += stress_len

            row.append(source)
            row.append(output)
            row.append(actual)
            row.append(str(we))
            row.append(str(word_error))
            row.append(str(syllable_error))
            row.append(str(stress_error))

            writer.writerow(row)

    performance = "Word_Error_Rate: "+ str(WER/total_words) + "\nPhoneme_Error_Rate: "+ str(PER/total_phones) + "\nSyllable_Error_Rate: "+ str(SyER/total_syllable) + "\nStress_Error_Rate: "+ str(SER/total_stress_marks)
    fw1.write(performance)

    fr0.close()
    fr1.close()
    fr2.close()
    fw.close()
    fw1.close()

if __name__ == '__main__':
    main()