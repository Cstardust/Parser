

rel_dct = {
    'root': '根节点',
    'sasubj-obj': '同主同宾',
    'sasubj': '同主语',
    'dfsubj': '不同主语',
    'subj': '主语',
    'subj-in': '内部主语',
    'obj': '宾语',
    'pred': '谓语',
    'att': '定语',
    'adv': '状语',
    'cmp': '补语',
    'coo': '并列',
    'pobj': '介宾',
    'iobj': '间宾',
    'de': '的',
    'adjct': '附加',
    'app': '称呼',
    'exp': '解释',
    'punc': '标点',
    'frag': '片段',
    'repet': '重复',
    # rst
    'attr': '归属',
    'bckg': '背景',
    'cause': '因果',
    'comp': '比较',
    'cond': '状况',
    'cont': '对比',
    'elbr': '阐述',
    'enbm': '目的',
    'eval': '评价',
    'expl': '解释-例证',
    'joint': '联合',
    'manner': '方式',
    'rstm': '重申',
    'temp': '时序',
    'tp-chg': '主题变更',
    'prob-sol': '问题-解决',
    'qst-ans': '疑问-回答',
    'stm-rsp': '陈述-回应',
    'req-proc': '需求-处理',
}

rel2id = {}
for i, (key, value) in enumerate(rel_dct.items()):
    rel2id[key] = i

id2rel = {}
for i, (key, value) in enumerate(rel_dct.items()):
    id2rel[i] = key

# print(rel2id)
# {'root': 0, 'sasubj-obj': 1, 'sasubj': 2, 'dfsubj': 3, 'subj': 4, 'subj-in': 5, 'obj': 6, 'pred': 7, 'att': 8, 'adv': 9, 'cmp': 10, 'coo': 11, 'pobj': 12, 'iobj': 13, 'de': 14, 'adjct': 15, 'app': 16, 'exp': 17, 'punc': 18, 'frag': 19, 'repet': 20, 'attr': 21, 'bckg': 22, 'cause': 23, 'comp': 24, 'cond': 25, 'cont': 26, 'elbr': 27, 'enbm': 28, 'eval': 29, 'expl': 30, 'joint': 31, 'manner': 32, 'rstm': 33, 'temp': 34, 'tp-chg': 35, 'prob-sol': 36, 'qst-ans': 37, 'stm-rsp': 38, 'req-proc': 39}
punct_lst = ['，', '.', '。', '!', '?', '~', '...', '......', ',', ':', '：', ';']