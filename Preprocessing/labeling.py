# -*- coding: utf-8 -*-


def Write2file(labels, reviews,index):
    """
    Write output.txt that suit for the code
    :param labels:  list F S A 1 2 3neutral
    :param reviews: list
    :param index: one number (it will increment)
    :return:
    """
    file_output = open("./Output.txt",'a')
    file_output_2 = open("./Output_FSA.txt",'a')
    for i in range(len(labels)):
        file_output.write("\t<sentence id=\"" + str(index) + "\">"+ "\n")
        file_output.write("\t\t<text>" + reviews[i] + "</text>" + "\n")
        file_output.write("\t\t<Opinions>"+ "\n")
        elements = labels[i].split()
        for j in range(len(elements)):
            aspect = list(elements[j]) #index 0 là aspect, index 1 là sentiment tương ứng
            cor_aspect_string = "" # chuỗi tương ứng để ghi file
            cor_sentiment_string = "" # chuỗi tương ứng để ghi file
            if(aspect[0] == 'F'):
                cor_aspect_string = 'FOOD'
            if(aspect[0] == 'S'):
                cor_aspect_string = 'STAFF'
            if(aspect[0] == 'A'):
                cor_aspect_string = 'AMBIENCE'
            if(aspect[0] == 'P'):
                cor_aspect_string = 'PRICE'
            if(aspect[0] == 'X'):
                cor_aspect_string = 'SERVICE'
            if(aspect[0] == 'O'):
                cor_aspect_string = 'OTHER'

            if(aspect[1] == '1'):
                cor_sentiment_string = 'positive'
            if(aspect[1] == '2'):
                cor_sentiment_string = 'negative'
            if(aspect[1] == '3'):
                cor_sentiment_string = 'neutral'

            file_output.write("\t\t\t<Opinion category=\"" + cor_aspect_string + "#GENERAL\" " + "polarity=\"" + cor_sentiment_string + "\" target=\"null\" to=\"null\"/>" + "\n")
        file_output.write("\t\t</Opinions>"+ "\n")
        file_output.write("\t</sentence>"+ "\n\n")

        if (len(labels[i])==2 and ('F' in labels[i] or 'S' in labels[i] or 'A' in labels[i]) and ('3' not in labels[i])): # Nếu có 1 nhãn duy nhất VÀ POS HOẶC NEG THÔI
            file_output_2.write("\t<sentence id=\"" + str(index) + "\">"+ "\n")
            file_output_2.write("\t\t<text>" + reviews[i] + "</text>"+ "\n")
            file_output_2.write("\t\t<Opinions>"+ "\n")
            aspect = list(labels[i]) #index 0 là aspect, index 1 là sentiment tương ứng
            cor_aspect_string = "" # chuỗi tương ứng để ghi file
            cor_sentiment_string = "" # chuỗi tương ứng để ghi file
            if(aspect[0] == 'F'):
                cor_aspect_string = 'FOOD'
            if(aspect[0] == 'S'):
                cor_aspect_string = 'STAFF'
            if(aspect[0] == 'A'):
                cor_aspect_string = 'AMBIENCE'
            if(aspect[0] == 'P'):
                cor_aspect_string = 'PRICE'
            if(aspect[0] == 'X'):
                cor_aspect_string = 'SERVICE'
            if(aspect[0] == 'O'):
                cor_aspect_string = 'OTHER'
            if(aspect[1] == '1'):
                cor_sentiment_string = 'positive'
            if(aspect[1] == '2'):
                cor_sentiment_string = 'negative'
            if(aspect[1] == '3'):
                cor_sentiment_string = 'neutral'
            file_output_2.write("\t\t\t<Opinion category=\"" + cor_aspect_string + "#GENERAL\" " + "polarity=\"" + cor_sentiment_string + "\" target=\"null\" to=\"null\"/>" + "\n")
            file_output_2.write("\t\t</Opinions>"+ "\n")
            file_output_2.write("\t</sentence>"+ "\n\n")
        index = index+1
    file_index = open("Index.txt",'w')
    file_index.write(str(index))