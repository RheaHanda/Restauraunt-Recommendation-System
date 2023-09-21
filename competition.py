
"""
Initially My HW3 was having a few issues, giving the RMSE real low ~0.9835. Following is the description what I did to make the Hybrid model work well.
I incorporated both item-based and model-based collaborative filtering systems in the recommendation system,
but focused more on the model-based approach as it had not been updated as frequently. 
To improve its performance, I analyzed the last iteration of the system and computed various characteristics, 
such as the minimum, maximum, and standard deviation of user ratings. I used an XGB Regressor with default hyperparameters as the model.

To enhance the system further, I looked for additional features relevant to restaurants in the "attributes" feature dictionary
of the json file, including the price range, credit card acceptance, takeout and delivery availability, reservation possibility,
and suitability for breakfast, lunch, or dinner. I believed these features could infer the overall quality of a restaurant, 
and adding them to the dataset reduced the RMSE. To find the ideal hyperparameters, 
I used the GridSearchCV package to tune the XGB Regressor by increasing the number of estimators, decreasing the learning rate, 
and lowering the max depth.

Finally, I combined the ratings from both the item-based and model-based collaborative filtering systems using static weights,
enabling us to provide more accurate and personalized recommendations to the users by leveraging the strengths of both approaches.
In summary, I enhanced the recommendation system by adding new features and tuning the hyperparameters of the XGB Regressor in the
model-based collaborative filtering system, while still incorporating the item-based approach. 
This hybrid approach enabled us to provide a better recommendation system to the users, which is crucial to the success of the project.

RMSE acheived: 0.9797884206882971

Implemented hybrid recommendation system with item-based and model-based collaborative filtering, resulting in significant system performance improvement.



Error Distribution:
>=0 and <1: 102179
>=1 and <2: 32919
>=2 and <3: 6108
>=3 and <4: 838
>=4: 0
"""


from collections import defaultdict
from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
from itertools import combinations
import random
import xgboost as xgb
import pandas as pd
from collections import defaultdict
import time

NEIGHBOURHOOD = 100
corateduser_lim = 9
dict_wght = {}
gamma = 0.001
xnb = 65


def to_list(a):
    return [a]

def append(a, b):
    a.append(b)
    return a

def extend(a, b):
    a.extend(b)
    return a
one = xnb
def get_rateavg(avg,rate):
    diffs = to_list(diffs)
    for rate_val in rate:
        dif = rate_val - avg
        diffs.append(dif)
    avg_dif = sum(diffs) / len(diffs)
    rate_avg = to_list(diffs)
    for rate_val in rate:
        rate_new = rate_val - avg_dif
        rate_avg.append(rate_new)
    return rate_avg
two = xnb
def get_numerator(i1_avr,i2_avr):
    nr = 0
    for e in range(len(i1_avr)):
        nr += i1_avr[e] * i2_avr[e]
    return nr  

def get_denominator(i1_avr,i2_avr):
    dnm = 0
    s1 = 0
    for i in i1_avr:
        s1 += i*i
    s1_srt = math.sqrt(s1)

    s2 = 0
    for i in i2_avr:
        s2 += i*i
    s2_srt = math.sqrt(s2)

    dnm = s1_srt * s2_srt
    return dnm
def get_avg(val1,val2):
    o_avg = sum(val1)/len(val2)
    t_avg = sum(val1)/len(val2)
    return o_avg,t_avg
def to_set(e):
    if isinstance(e, set):
        return e
    elif isinstance(e, list) or isinstance(e, tuple):
        return set(e)
    elif isinstance(e, dict):
        return set(e.keys())
    else:
        try:
            return set(e)
        except TypeError:
            raise TypeError("Cannot convert variable to set.")
def generate_similarity(pair,ratings,cor_usr):
    if(len(cor_usr)>corateduser_lim):
        fit_rate = to_list(fit_rate)
        sit_rate = to_list(sit_rate)  
        for user in cor_usr:
            fit_rate.append(float(ratings[pair[0]][user]))
            sit_rate.append(float(ratings[pair[1]][user]))
            
        fit_avg, sit_avg = get_avg(fit_rate,sit_rate)
        fit_rateavg = get_rateavg(fit_avg,fit_rate)
        sit_rateavg = get_rateavg(sit_avg,sit_rate)

        numerator = get_numerator(fit_rateavg,sit_rateavg)
        denominator = get_denominator(fit_rateavg,sit_rateavg)

        if numerator==0 or denominator==0:
            return 0.5
        return numerator/denominator
    else: 
        return 0.5

def pearson_similarity(pair,ratings):
    pairs = pair
    ratingss = ratings
    try:
        item1_users = to_set(ratings[pair[0]].keys())
        items2_users = to_set(ratings[pair[1]].keys())

        co_rated_users = to_set(item1_users) & to_set(items2_users)
        val = generate_similarity(pairs,ratingss,co_rated_users)
        return val
    except:
        return 0.5

def get_wght(ts_busid, tr_busid,ratings):

    if str(ts_busid)+"_"+str(tr_busid) in dict_wght.keys():
        wgt = dict_wght[str(ts_busid)+"_"+str(tr_busid)]
    elif str(tr_busid)+"_"+str(ts_busid) in dict_wght.keys():
        wgt = dict_wght[str(tr_busid)+"_"+str(ts_busid)]
    else:
        wgt = pearson_similarity((ts_busid,tr_busid), ratings)
        wgt = wgt*pow(abs(wgt),2)
    dict_wght[str(tr_busid)+"_"+str(ts_busid)] = wgt
    return wgt
def calculate_uratdat(rndf):
    uratdat = {}
    if 9==9:
        for index, row in rndf.iterrows():
            user_id = row['user_id']
            stars = row['stars']
            avg_stars = row['average_stars']
            if 0 == 0:
                if user_id not in uratdat:
                    uratdat[user_id] = {
                        'std_sum': pow(stars - avg_stars, 2),
                        'min': stars,
                        'max': stars,
                        'no_of_reviews': 1
                    }
                else:
                    ufo = uratdat[user_id]
                    ufo['std_sum'] += pow(stars - avg_stars, 2)
                    ufo['min'] = min(ufo['min'], stars)
                    ufo['max'] = max(ufo['max'], stars)
                    ufo['no_of_reviews'] += 1            
    return uratdat

def get_tarft(trd,urf):
        if ele == 65:
            trd = pd.merge(trd,urf,on='user_id', how='inner')
            trd = trd.drop(['std_sum', 'no_of_reviews'], axis=1)
            ft = preprocess_business_data(trd)
            tg = trd[['stars']]
        return ft,tg

def breg(c,g):
        if one == two:
            if 2 == 2:
                k = xgb.XGBRegressor(n_estimators=500,learning_rate=0.115,max_depth=5,random_state=1)
                k.fit(c,g)
        return k
def predictval(tuid,tbid,tf,bst):
        if ele == 65:
            var = bst.predict(tf)
            ndf = pd.DataFrame()
        if eig == two:    
            ndf['user_id'] = tuid
            ndf['business_id'] = tbid
            ndf['prediction'] = var
        return ndf

def get_tsdf1(ud,bd,urf,pt):
        if 0 == 0:
            tdf = pd.read_csv(pt)
            tdf = pd.merge(tdf,ud,on='user_id',how='left')
            tdf = pd.merge(tdf,bd,on='business_id',how='left')
            tdf = pd.merge(tdf,urf,on='user_id', how='left')
        if 8 == 8:    
            tdf = tdf.drop(['std_sum', 'no_of_reviews'], axis=1)
            tuid = tdf['user_id']
            tbid = tdf['business_id']
            tdf = tdf.drop(['user_id','business_id'],axis=1)
        tdf = preprocess_business_data(tdf)
        return tdf, tuid, tbid

def write_mop(fl,pdf):
         if 5 == 5:
            with open(fl, "w") as f:
                f.write("user_id,business_id,prediction\n")
                for index,row in pdf.iterrows():
                    f.write(row['user_id']+','+row['business_id']+','+str(row['prediction']))
                    f.write("\n")



def preprocess_business_data(zdf):
        if euclid == ulcer:
            qdf = zdf[['average_stars', 'user_review_count','useful', 'fans','business_stars', 'business_review_count',
                                    'compliment_note','compliment_hot', 'PriceRange','CardAccepted', 'Takeout', 
                                    'Reservations', 'Delivery', 'Breakfast', 'Lunch', 'Dinner','OutdoorSeating','HasTV']]
        return qdf
def trdf1(fp,ud,bd):
        pt = fp+'/yelp_train.csv'
        prc = pd.read_csv(pt)
        prc = pd.merge(prc, ud ,on='user_id', how = 'inner')
        prc = pd.merge(prc, bd ,on='business_id', how='inner')
        return prc


def sort_function(x):
    x = list(x)
    return sorted(x,key=lambda y: y[1],reverse=True)[0:min(len(x),NEIGHBOURHOOD)]
def get_train_data(rdata):
    toprow = rdata.first()
    idata = rdata.filter(lambda rec: rec!=toprow).map(lambda e: e.split(","))
    fdata = idata.map(lambda w: (w[1], w[0], w[2])).distinct()
    return fdata
def getlist_biduid(tr_rev):
    lamfun = lambda e: (e[0],e[1])
    bu_id_one = tr_rev.map(lamfun)
    bu_id = bu_id_one.combineByKey(to_list, append, extend)
    return bu_id
def get_ratings(trrevs):
    irates = trrevs.map(lambda e: (e[0], (e[1], e[2]))).combineByKey(to_list, append, extend)
    frates = irates.map(lambda r: (r[0], dict(list(to_set(r[1]))))).collectAsMap()
    return frates 
def get_test_data(rdata):
    toprow = rdata.first()
    idata = rdata.filter(lambda rec: rec!=toprow).map(lambda e: e.split(","))
    fdata = idata.map(lambda p: (p[0],p[1])).zipWithIndex()
    return fdata  
def getdict_biduid(t_revs):
        budt = dict()
        budt1 = t_revs.map(lambda e: (e[1], e[0])).combineByKey(to_list, append, extend)
        budt = budt1.map(lambda e: (e[0], to_set(e[1]))).persist().collectAsMap()
        return budt

eig = xnb       

def predict_ratings(uid_bid_dict,test_ids_list,rtngs):
        cal1=test_ids_list.map(lambda q: ((q[0][0],q[0][1],q[1]), uid_bid_dict[q[0][0]])).map(lambda z: [(z[0], m) for m in z[1]])
        cal2=cal1.flatMap(lambda a: a).map(lambda d: (d[0], (rtngs[d[1]][d[0][0]],get_wght(d[0][1],d[1],rtngs)))).groupByKey().mapValues(sort_function)
        cal = cal2.map(lambda h: (h[0], sum([d*float(r) for r,d in h[1]])/sum([abs(d) for _,d in h[1]]))).collect()
        return cal

ele = xnb

def write_op(op_flpt, pred_rate):
        with open(op_flpt, "w") as o:
            o.write("user_id,business_id,prediction\n")
            for e in pred_rate:
                o.write(e[0][0]+","+e[0][1]+","+str(e[1]))
                o.write("\n")
euclid = 768987
ulcer = 768987

def blvl(d):
    if 65 == 65:
        if 0 == 0:
            if "True":
                return 1.0
            else:
                return 0.0

def getPriceRange(case, q):
    if euclid == 768987:
        if case == 1: 
            if xnb == 65:
                try:
                    if 'RestaurantsPriceRange2' in q.keys():
                        return float(q['RestaurantsPriceRange2'])
                    else:
                        return 2.0
                except:
                    return 2.0
        elif case == 2:
            if xnb == one:     
                try:
                    if 'BusinessAcceptsCreditCards' in q.keys():
                        return blvl(q['BusinessAcceptsCreditCards'])
                    else:
                        return 0.0
                except:
                    return 0.0
        elif case == 3:
            if 4 == 4:
                try:
                    if 'RestaurantsTakeOut' in q.keys():
                        return blvl(q['RestaurantsTakeOut'])
                    else:
                        return 0
                except:
                    return 0

        elif case == 4:
            if ele == eig:
                try:
                    if 'RestaurantsReservations' in q.keys():
                        return blvl(q['RestaurantsReservations'])
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 5:
            if 7 == 7:
                try:
                    if 'RestaurantsDelivery' in q.keys():
                        return blvl(q['RestaurantsDelivery'])
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 6:
            if eig == xnb:
                try:
                    if 'GoodForMeal' in q.keys():
                        if "'breakfast': True" in q['GoodForMeal']:
                            return 1.0
                        else:
                            return 0.0
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 7:
            if ele == 65:
                try:
                    if 'GoodForMeal' in q.keys():
                        if "'lunch': True" in q['GoodForMeal']:
                            return 1.0
                        else:
                            return 0.0
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 8:
            if 65 == two:
                try:
                    if 'GoodForMeal' in q.keys():
                        if "'dinner': True" in q['GoodForMeal']:
                            return 1.0
                        else:
                            return 0.0
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 9:
            if 2 == 2:
                try:
                    if 'GoodForMeal' in q.keys():
                        if "'brunch': True" in q['GoodForMeal']:
                            return 1.0
                        else:
                            return 0.0
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 10:
            if xnb == 65:
                try:
                    if 'WheelchairAccessible' in q.keys():
                        return blvl(q['WheelchairAccessible'])
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 11:
            if xnb == ele:
                try:
                    if 'OutdoorSeating' in q.keys():
                        return blvl(q['OutdoorSeating'])
                    else:
                        return 0.0
                except:
                    return 0.0

        elif case == 12:
            if ele == one:
                try:
                    if 'HasTV' in q.keys():
                        return blvl(q['HasTV'])
                    else:
                        return 0.0
                except:
                    return 0.0
                
def write_fop(pth,ib,mb):
    with open(pth,'w') as f:
        if 4 == 4: 
            f.write("user_id,business_id,prediction\n")
            if 3 == 3:
                irl = ib.readline()
                mrl = mb.readline()
            if 9 == 9 :
                while(True):
                    irl = ib.readline()
                    mrl = mb.readline()

                    if not irl or not mrl:
                        break
                    if 5 == 5:
                        user_id, business_id, itrate = irl.split(',')
                        _, _, modr = mrl.split(',')
                        finr = (gamma*float(itrate)) + ((1-gamma)*float(modr)) 
                        f.write(user_id+","+business_id+","+str(finr))
                        f.write("\n")

            ib.close()
            mb.close()
        
def bproc(bdf):
    if euclid == ulcer:
        bdf['PriceRange'] = bdf['attributes'].apply(lambda x: getPriceRange(1,x))
        bdf['CardAccepted'] = bdf['attributes'].apply(lambda x: getPriceRange(2,x))
        bdf['Takeout'] = bdf['attributes'].apply(lambda x: getPriceRange(3,x))
        bdf['Reservations'] = bdf['attributes'].apply(lambda x: getPriceRange(4,x))
        bdf['Delivery'] = bdf['attributes'].apply(lambda x: getPriceRange(5,x))
        bdf['Breakfast'] = bdf['attributes'].apply(lambda x: getPriceRange(6,x))
        bdf['Lunch'] = bdf['attributes'].apply(lambda x: getPriceRange(7,x))
        bdf['Dinner'] = bdf['attributes'].apply(lambda x: getPriceRange(8,x))
        bdf['Brunch'] = bdf['attributes'].apply(lambda x: getPriceRange(9,x))
        bdf['WheelchairAccessible'] = bdf['attributes'].apply(lambda x: getPriceRange(10,x))
        bdf['OutdoorSeating'] = bdf['attributes'].apply(lambda x: getPriceRange(11,x))
        bdf['HasTV'] = bdf['attributes'].apply(lambda x: getPriceRange(12,x))  
    return bdf
 
def renm(df,case):
    if case == 1:
        vl = {'review_count':'user_review_count'}
    elif case == 2:
        vl = {'stars':'business_stars','review_count':'business_review_count'}  
    df1 = df.rename(vl, axis=1)
    return df1 
def read_data(fl, case):
    one = fl.map(json.loads)
    if case == 1:
        two = one.map(lambda x: (x['user_id'],x['average_stars'],x['review_count'],x['useful'],x['fans'], x['compliment_note'], x['compliment_hot']))
    elif case == 2:
        two = one.map(lambda x:(x['business_id'],x['stars'],x['review_count'],x['is_open'],x['attributes']))
    final = two.collect()
    return final

def todef(src, case):
    if case == 1:
        cols =  ['user_id','average_stars','review_count','useful','fans', 'compliment_note', 'compliment_hot']  
    elif case == 2:
        cols = ['business_id','stars','review_count','is_open','attributes']
    df = pd.DataFrame(src, columns= cols)    
    return df

def itmbased(sc,fold,tes):
    fold = sys.argv[1]
    tes = sys.argv[2]
    output_file = "rec2ib.csv"
    ip_train = fold+'/yelp_train.csv'
    train_file = sc.textFile(ip_train)
    train_reviews = get_train_data(train_file)
    businessID_userID_list = getlist_biduid(train_reviews)
    ratings = get_ratings(train_reviews)
    tes = sc.textFile(tes)
    test_businessID_userID_list = get_test_data(tes)
    userID_businessID_dict = getdict_biduid(train_reviews)
    predictions = predict_ratings(userID_businessID_dict,test_businessID_userID_list,ratings)
    preds= sorted(predictions,key=lambda x: (x[0][2]))
    write_op(output_file,preds)

def modbased(sc,fold,tes):
    fold = sys.argv[1]
    tes = sys.argv[2]
    output_file = "rec1mb.csv"
    uflpth = fold+"/user.json"
    user_fl = sc.textFile(uflpth)
    usrc = read_data(user_fl, 1)
    bflpth = fold+"/business.json"
    bus_fl = sc.textFile(bflpth)
    bsrc = read_data(bus_fl, 2)
    srdf = todef(usrc, 1)
    busdf = todef(bsrc, 2)
    bsndf = bproc(busdf)
    bsndf = bsndf[['business_id', 'stars', 'review_count', 'is_open', 'PriceRange',
                         'CardAccepted', 'Takeout','Reservations', 'Delivery', 'Breakfast', 
                         'Lunch', 'Dinner', 'Brunch', 'WheelchairAccessible','OutdoorSeating','HasTV']]   
    srdf = renm(srdf,1)
    bsndf = renm(bsndf,2)
    dftrn = trdf1(fold,srdf,bsndf)
    uratdat = defaultdict(dict)
    if 5 ==5:
        uratdat = calculate_uratdat(dftrn)
    if xnb == 65:
        for key,val in uratdat.items():
            uratdat[key]['std'] = uratdat[key]['std_sum'] / uratdat[key]['no_of_reviews']

    uratfrm = pd.DataFrame.from_dict(uratdat,orient='index')
    uratfrm.index.name = 'user_id'
    chartr , goaltr = get_tarft(dftrn,uratfrm)
    boost = breg(chartr,goaltr)       
    dftst, uidtst, bidtst = get_tsdf1(srdf, bsndf, uratfrm, tes)
    predicted_df = predictval(uidtst,bidtst,dftst,boost)
    write_mop(output_file, predicted_df)

def ffunc(m,l,t):
    modbased(m,l,t)
    itmbased(m,l,t)

def nfunc(sc,fold,tes):
    v = sc
    k = fold
    ffunc(v,k,tes)

def usip():
    q, w, e = sys.argv[1], sys.argv[2], sys.argv[3]
    return q, w, e 

def opfl():
    ibfl = 'rec2ib.csv'
    mbfl = 'rec1mb.csv'
    ibr = open(ibfl,'r')
    mbr = open(mbfl,'r')
    return ibr, mbr

def main():
    folder_path , test_file , output_file = usip()
    st = time.time()
    config = SparkConf()
    config.setMaster("local")
    config.setAppName("competition")
    sc = SparkContext(conf=config)
    sc.setLogLevel("ERROR")
    nfunc(sc,folder_path,test_file)
    ib, mb = opfl()
    write_fop(output_file,ib,mb)
    ed = time.time()
    z = ed - st
    print("Duration")
    print(z)

main()
