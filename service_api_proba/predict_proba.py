import datetime

import pymongo
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
import pandas as pd
from dateutil.parser import parse
from model.db_init import db_init


def group_cibil_score(latest_cibil_score):
    # =IF(C2 <= 300, "LTE_300", IF(C2 <= 400, "300 TO 400", IF(C2 <= 500, "400 TO 500", IF(C2 <= 600, "500 TO 600",
    #   IF(C2 <= 700, "600 TO 700", IF(C2 <= 800, "700 TO 800",IF(C2 <= 900,"800 TO 900",C2)))))))

    int_latest_cibil_score = int(latest_cibil_score)
    if "-" in str(latest_cibil_score):
        data_latest_cibil_score = -1
    elif 0 <= int_latest_cibil_score <= 300:
        data_latest_cibil_score = 3
    elif 0 <= int_latest_cibil_score <= 400:
        data_latest_cibil_score = 4
    elif 0 <= int_latest_cibil_score <= 500:
        data_latest_cibil_score = 5
    elif 0 <= int_latest_cibil_score <= 600:
        data_latest_cibil_score = 6
    elif 0 <= int_latest_cibil_score <= 700:
        data_latest_cibil_score = 7
    elif 0 <= int_latest_cibil_score <= 800:
        data_latest_cibil_score = 8
    elif 0 <= int_latest_cibil_score <= 900:
        data_latest_cibil_score = 9

    return data_latest_cibil_score


def predict_default(business_id):
    db = db_init()

    # scf invoice get business_id for that invoice
    cas_business = db['cas_business']
    data = cas_business.find({"business_id": business_id}, no_cursor_timeout=True).sort("_id",
                                                                                        pymongo.DESCENDING).limit(1)
    for i, row in enumerate(data):
        date_of_birth = str(row['business_partners'][0]['date_of_birth'])
        print("date_of_birth : " + str(date_of_birth))
        latest_cibil_score = row['latest_cibil_score']

    date_of_birth = parse(date_of_birth)
    print(date_of_birth)
    now = datetime.datetime.now()
    print(now)

    age = now - date_of_birth
    print(age.days)
    from datetime import date, timedelta

    age = age.days / 365.2425
    print(int(age))

    # 1. get age from 0th business partner - cas business
    # 2. latest_cibil_score - cas business
    print(latest_cibil_score)
    cibil_score_group = group_cibil_score(latest_cibil_score)

    # scf invoice get business_id for that invoice
    lms_limits_master = db['lms_limits_master']
    data = lms_limits_master.find({"business_id": business_id}, no_cursor_timeout=True).sort("_id",
                                                                                             pymongo.DESCENDING).limit(
        1)
    for i, row in enumerate(data):
        limit_amount_sanctioned = str(row['limit_amount_sanctioned'])
        no_credit_bounces = row['no_credit_bounces']
        no_credit_bounces_last_6_months = row['no_credit_bounces_last_6_months']
        no_credit_bounces_last_3_months = row['no_credit_bounces_last_3_months']
        back_to_back_bounces_last_3_months = row['back_to_back_bounces_last_3_months']

    lms_limit_master = ["limit_amount_sanctioned",
                        "no_credit_bounces",
                        "no_credit_bounces_last_6_months",
                        "no_credit_bounces_last_3_months",
                        "back_to_back_bounces_last_3_months"]

    cas_business = ["age", "latest_cibil_score"]

    cols = lms_limit_master.append(cas_business)
    all_data_df = pd.DataFrame(columns=cols)

    all_data_df.loc[0, 'cibil_score_group'] = cibil_score_group
    all_data_df.loc[0, 'age'] = age
    all_data_df.loc[0, 'limit_amount_sanctioned'] = limit_amount_sanctioned
    all_data_df.loc[0, 'no_credit_bounces'] = no_credit_bounces
    all_data_df.loc[0, 'no_credit_bounces_last_6_months'] = no_credit_bounces_last_6_months
    all_data_df.loc[0, 'no_credit_bounces_last_3_months'] = no_credit_bounces_last_3_months
    all_data_df.loc[0, 'back_to_back_bounces_last_3_months'] = back_to_back_bounces_last_3_months

    all_data_df = pd.concat(
        [all_data_df, pd.get_dummies(all_data_df['cibil_score_group'], prefix='category', dummy_na=True)], axis=1).drop(
        ['cibil_score_group'], axis=1)

    # print(all_data_df)

    # load and predict
    all_cols = ["limit_amount_sanctioned",
                "no_credit_bounces",
                "age",
                "no_credit_bounces_last_6_months",
                "category_8.0",
                "no_credit_bounces_last_3_months",
                "category_-1.0",
                "back_to_back_bounces_last_3_months",
                "category_6.0"]

    if cibil_score_group == 8:
        all_data_df['category_6.0'] = 0
        all_data_df['category_-1.0'] = 0

    elif cibil_score_group == -1:

        all_data_df['category_6.0'] = 0
        all_data_df['category_8.0'] = 0

    elif cibil_score_group == 6:

        all_data_df['category_8.0'] = 0
        all_data_df['category_-1.0'] = 0
    else:
        all_data_df['category_8.0'] = 0
        all_data_df['category_6.0'] = 0
        all_data_df['category_-1.0'] = 0

    predict_df = all_data_df[all_cols]
    print(predict_df)

    import pickle

    filename = 'service_api_proba/proba_8_cols.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    result = loaded_model.predict_proba(predict_df)
    print(result)
    print("Probablity of giving : " + str(result[0][1]))
    response = str(result[0][1])
    return response
    # predict_df['output_predicted'] = result
    #
    # predict_df.to_csv('/Users/admin-mac/PycharmProjects/data_credit_default/service_api_default_predict_model/output_decision_tree.csv')
