import re
import os
import json
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from PyKomoran import Komoran
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from util.dbutil import db_connector
from textrank.summarizer import KeywordSummarizer


# 통문장 형태 텍스트를 한 문장 리스트로 생성
def fn_text_split_sentence(full_text):

    list_temp = re.sub(r'([^\n\s\.\?!]+[^\n\.\?!]*[\.\?!])', r'\1\n', full_text).strip().split("\n")
    list_sentence = []   

    for idx, row in enumerate(list_temp):
        sentence = row.strip()

        if len(sentence) != 0:
            list_sentence.append(sentence)

    return list_sentence


# 리스트형태의 텍스트를 한 문장 리스트로 생성
def fn_list_split_sentece(origin_text, split_type):
    list_sentence = []

    if split_type == 'instroy':
        origin_text = origin_text.replace("['", '').replace("']", '').replace('\\n\\n', '\n')
        list_sentence = fn_text_split_sentence(origin_text)

    elif split_type == 'recomendation':
    
        origin_text = origin_text.replace("['", '').replace("']", '')
        list_origin = origin_text.split("', '")

        full_text = ''
        for row in list_origin:
            if len(row.split(':', 1)) == 2:
                temp_text = row.split(':', 1)[1].strip()
                full_text += temp_text
        
        list_sentence = fn_text_split_sentence(full_text)
        
    return list_sentence


# 명사 문장 만들기 (리스트 변환)
def fn_make_noun_sentences(model_komoran, list_sentence, data_type):
    result = []

    # 불용어 정리
    list_stopword = ['질문', '사이', '관계', '출판']

    if data_type == 'content':
        list_stopword = ['옮긴이', '글쓴이', '제목', '목차', '미주', '머리말']
    
    for sentence in list_sentence:
        list_token = model_komoran.get_token_list(sentence)

        list_word = []
        list_compound = []
        
        for idx, token in enumerate(list_token):
            
            text = token.morph
            pos = token.pos

            index_begin = token.begin_index
            index_end = token.end_index
            
            if pos == 'NNP' or pos == 'NNG':
                if len(text) > 1:
                    list_word.append('{}'.format(text))                            

            if idx > 0:
                # 'NNP', 'NNG'
                check_num = 0
                check_pos = list_token[idx-1].pos
                check_end = list_token[idx-1].end_index
                
                if check_pos == 'NNP' or check_pos == 'NNG':
                    check_num += 1

                if pos == 'NNP' or pos == 'NNG':
                    check_num += 1
                
                if check_num == 2:
                    before_text = list_token[idx-1].morph

                    if check_end == index_begin:
                        compound_word = before_text + text
                        list_compound.append('{}'.format(compound_word))
                    elif check_end + 1 == index_begin:
                        compound_word = before_text + '_' + text
                        list_compound.append('{}'.format(compound_word))

        # 불용어 처리
        for stopword in list_stopword:
            list_word = [value for value in list_word if value != stopword]

        temp_one = ' '.join(list_word)
        temp_two = ' '.join(list_compound)
        temp_total = temp_one + ' ' + temp_two
        temp_total = temp_total.strip()
        
        if len(temp_total) > 0:
            # temp_total = temp_total[:-1]
            result.append(temp_total)

    return result

## TF-IDF 함수
def fn_tf_idf_keyword(list_sentence):

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(list_sentence)

    # TF-IDF 행렬에서 가장 높은 값을 갖는 단어 추출
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray().T.max(axis=1)
    tf_idf_keyword = sorted([(scores[i], feature_names[i]) for i in range(len(scores))], reverse=True)

    return tf_idf_keyword
"""
# 키워드 랭킹 
def fn_word_rank(list_sentence):

    # KwordRank 
    min_count = 2   # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 20 # 단어의 최대 길이
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=False)
    beta = 0.85    # PageRank의 decaying factor beta
    max_iter = 20
    
    keywords, rank, graph = wordrank_extractor.extract(list_sentence, beta, max_iter)

    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True):
        print('%8s:\t%.4f' % (word, r))

    return keywords
"""


if __name__=="__main__":
    
    # 출판사 별로 각각 실행
    print("키워드 추출 (교보문고) + Komoran 활용 >>>>> >>>>> >>>>> >>>>> >>>>>")    

    # KOMORAN 객체 생성
    komoran = Komoran("STABLE")
    
    # DB 연결
    conn = db_connector()
    cur = conn.cursor()

    column_list = ['id', 'collect_id', 'book_title', 'book_subtitle', 'book_origin_title', 'isbn', 
                   'book_topic', 'author_list', 'publish_date', 'book_introduction', 'book_content', 
                   'book_instroy', 'publisher_review', 'publisher_book_intro', 'book_recomendation', 'book_keyword']
    
    sql_select_book = """ SELECT id, collect_id, book_title, book_subtitle, book_origin_title, isbn, 
                                 book_topic, author_list, publish_date, book_introduction, book_content, 
                                 book_instroy, publisher_review, publisher_book_intro, book_recomendation, book_keyword
                            FROM book
                           WHERE collect_id = 1
                      """

    cur.execute(sql_select_book)

    list_book = cur.fetchall()

    # for row in tqdm(list_book):
    for row in list_book:
        list_content = list(row)
        
        book_id = list_content[0]
        isbn = list_content[5]
        book_title = list_content[2]
        book_subtitle = list_content[3]
        book_origin_title = list_content[4]
        keyword_basic = list_content[15]

        save_book_id = {
            'c_1_book_id': book_id,
            'keyword_extract': 'TF-IDF'
        }

        print('book_id = {}, {}'.format(book_id, book_title))

        book_topic = list_content[6]
        book_topic = book_topic.replace("['", '').replace("']", '')
        list_topic = book_topic.split("', '")

        save_topic_list = []
        
        # 주제 정리
        for topic in list_topic:
            list_one = topic.split('>')
            one_dict = {}
            for idx, t_value in enumerate(list_one):
                # 국내 도서 삭제
                if idx != 0:
                    t_value = t_value.strip()
                    dic_key = 'topic_{}'.format(idx)
                    one_dict[dic_key] = t_value

            save_topic_list.append(one_dict)
        
        # print('===================================================================')
        book_introduction = list_content[9]             # 책 소개
        book_content = list_content[10]                 # 목차
        book_instroy = list_content[11]                 # 책 속으로 (리스트 형태)
        publisher_review = list_content[12]             # 출판사 리뷰
        book_recomendation = list_content[14]           # 추천글 (리스트 형태)

        list_sentence_introduction = []
        list_sentence_content = []
        list_sentence_instroy = []
        list_sentece_review = []
        list_sentence_recomendation = []

        list_noun_sentence_introduction = []
        list_noun_sentence_content = []
        list_noun_sentence_instroy = []
        list_noun_sentence_review = []
        list_noun_sentence_recomendation = []
        
        # print('===================================================================')
        list_keyword_introduction = np.NaN
        list_keyword_content = np.NaN
        list_keyword_instroy = np.NaN
        list_keyword_review = np.NaN
        list_keyword_recomendation = np.NaN
        # print('===================================================================')

        # Textrank 객체 생성
        keyword_extractor_v1 = KeywordSummarizer(
            tokenize = lambda x:x.split(),      # YOUR TOKENIZER
            window = -1,
            min_count = 1,
            min_cooccurrence = 1,
            verbose = False,
            data_type = 'general'
        )

        # Textrank 객체 생성
        keyword_extractor_v2 = KeywordSummarizer(
            tokenize = lambda x:x.split(),      # YOUR TOKENIZER
            window = -1,
            min_count = 1,
            min_cooccurrence = 1,
            verbose = False,
            data_type = 'content'
        )

        # Textrank 객체 생성
        keyword_extractor_v3 = KeywordSummarizer(
            tokenize = lambda x:x.split(),      # YOUR TOKENIZER
            window = -1,
            min_count = 1,
            min_cooccurrence = 1,
            verbose = False,
            data_type = 'general'
        )

        # 텍스트로 되어 있는 덩어리 문장을 하나의 문장 단위 리스트로 만들기
        if book_introduction != '':
            list_sentence_introduction = fn_text_split_sentence(book_introduction)
            list_noun_sentence_introduction = fn_make_noun_sentences(komoran, list_sentence_introduction, 'general')

            if len(list_noun_sentence_introduction) > 1:
                list_keyword_introduction = fn_tf_idf_keyword(list_noun_sentence_introduction)
            # print('===================================================================')
        
        if book_content != '':
            list_sentence_content = fn_text_split_sentence(book_content)
            list_noun_sentence_content = fn_make_noun_sentences(komoran, list_sentence_content, 'content')

            if len(list_noun_sentence_content) > 1:
                list_keyword_content = fn_tf_idf_keyword(list_noun_sentence_content)
            # print('===================================================================')
            
        if book_instroy != "['']" and book_instroy != '[]':
            list_sentence_instroy = fn_list_split_sentece(book_instroy, 'instroy')
            list_noun_sentence_instroy = fn_make_noun_sentences(komoran, list_sentence_instroy, 'general')
            
            if len(list_noun_sentence_instroy) > 1:
                list_keyword_instroy = fn_tf_idf_keyword(list_noun_sentence_instroy)
            # print('===================================================================')

        if publisher_review != '':
            list_sentece_review = fn_text_split_sentence(publisher_review)
            list_noun_sentence_review = fn_make_noun_sentences(komoran, list_sentece_review, 'general')
            if len(list_noun_sentence_review) > 1:
                list_keyword_review = fn_tf_idf_keyword(list_noun_sentence_review)
            # print('===================================================================')

        if book_recomendation != "['']" and book_recomendation != "[]":
            list_sentence_recomendation = fn_list_split_sentece(book_recomendation, 'recomendation')        
            list_noun_sentence_recomendation = fn_make_noun_sentences(komoran, list_sentence_recomendation, 'general')
            
            if len(list_noun_sentence_recomendation) > 1:
                list_keyword_recomendation = fn_tf_idf_keyword(list_noun_sentence_recomendation)
            # print('===================================================================')

        # 목차 제외하고 텍스트 분석 한 부분 저장
        save_keyword = {
            'c_1_introduction': list_keyword_introduction,
            'c_1_content': list_keyword_content,
            'c_1_instroy': list_keyword_instroy,
            'c_1_reivew': list_keyword_review,
            'c_1_recomendation': list_keyword_recomendation
        }

        insert_keyword_sql =  """
            INSERT INTO book_keyword 
            (isbn, list_book_id, book_title, book_subtitle, book_origin_title, book_topic, keyword_basic, keyword_making, create_date)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, current_timestamp())       
        """
        insert_keyword_values = (isbn, save_book_id, book_title, book_subtitle, book_origin_title, save_topic_list, keyword_basic, save_keyword)
        # cur.execute(insert_keyword_sql, insert_keyword_values)
        # conn.commit()
        # time.sleep(0.07)

        # print('-----------------------------------------------')
        
    
    conn.close() 
    print('---------------------------------------------------------------------------------------------')
    print('------- SUCESS ------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------')