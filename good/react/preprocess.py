#!/usr/bin/env python3

import pandas as pd
import os
import pickle
import itertools

file_name = 'records.pickle'

# creates a pickled dictionary of records
def _to_pickle(file_name):
    files = [ x for x in sorted(os.listdir()) if x[0]=='d']

    dfs = [pd.read_csv(f,sep=';',quotechar='"') for f in files]

    dfs = pd.concat(dfs)

    themes = pd.read_csv("themes.csv",sep=';',quotechar='"')

    res = dfs.merge(themes,how='left',on='id_theme',sort=True).sort_values('title')

    res = res.to_dict('records')

    with open(file_name ,'wb') as h:
        pickle.dump(res,h,protocol=pickle.HIGHEST_PROTOCOL)


def get_main_comments(data):
    filtered=[]
    comments = [ list(g) for k,g in itertools.groupby(data,lambda x:x['title'])]
    for prispevok in comments:
        x =_clean_reactions(prispevok)
        filtered.append(x)

    return filtered

def _clean_reactions(l):
    new_list =[]
    uniqs = set([v['id_reaction'] for v in l])
    for x in l:
        if x['id_parent_reaction'] not in uniqs:
            new_list.append(x)
    return new_list



if __name__ == "__main__":
    # _to_pickle('records.pickle')

    with open("records.pickle",'rb') as h:
        data = pickle.load(h)

    x = get_main_comments(data)
    print(len(x))


    with open("half_filtered",'wb') as h:
        pickle.dump(x,h,protocol=pickle.HIGHEST_PROTOCOL)
