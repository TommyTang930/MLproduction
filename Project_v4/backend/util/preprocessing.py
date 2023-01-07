import json

def separate_id_col(h2o_frame):
    '''
    将id列分离出来，返回没有id列的数据
    :param h2o_frame: h2o frame格式的数据
    :return:
        id_name: ID列的名称
        X_id: ID这一列的数据
        X_rest: 除了ID列，剩余列的数据
    '''
    id_names = ['ID','Id','iD','id'] # ID列的名称的几种可能写法
    for i in id_names:
        if i in h2o_frame.names:
            id_name = i  # id列的名称
            X_id = h2o_frame[:, id_name] # id列的数据
            X_rest = h2o_frame.drop(id_name) # 其它列的数据
            break
        else:
            id_name, X_id = None, None
            X_rest = h2o_frame

    return id_name, X_id, X_rest


def match_col_types(h2o_frame):
    '''
    :param h2o_frame: 输入的test数据集
    :return:
    '''
    # 加载训练集中列的名称json文件
    with open("data/train_col_types.json", "r") as file:
        train_col_types = json.load(file)

    # 将test的列名与train的列名匹配/一致
    for key in train_col_types.keys():
        try:
            if train_col_types[key] != h2o_frame.types[key]:
                if train_col_types[key] == 'real' and h2o_frame.types[key] == 'enum':
                    h2o_frame[key] = h2o_frame[key].ascharacter().asnumeric()
                elif train_col_types[key] == 'real':
                    h2o_frame[key] = h2o_frame[key].asnumeric()
                elif train_col_types[key] == 'int':
                    h2o_frame[key] = h2o_frame[key].asfactor()
                elif train_col_types[key] == 'str':
                    h2o_frame[key] = h2o_frame[key].ascharacter()
        except:
            pass

    return h2o_frame