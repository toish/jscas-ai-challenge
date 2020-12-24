import pandas as pd
import numpy as np
import os
import cv2

# ディレクトリ内の画像を読み込む
# inputpath: ディレクトリのパス,  type_color: ColorかGray
def load_images(inputpath, type_color):
    imglist = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue
            if 'checkpoint' in fn:
                continue

            filename = os.path.join(root, fn)

            if type_color == 'Color':
                # カラー画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                # サイズ変更
#                 height, width = testimage.shape[:2]
#                 # 主に縮小するのでINTER_AREA使用
#                 testimage = cv2.resize(
#                     testimage, imagesize, interpolation=cv2.INTER_AREA)
#                 testimage = np.asarray(testimage, dtype=np.float32)
#                 # 色チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
#                 #testimage = testimage.transpose(2, 0, 1)
#                 # チャンネルをbgrからrgbの順に変更
#                 testimage = testimage[:, :, ::-1]

            elif type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # サイズ変更
#                 height, width = testimage.shape[:2]
#                 # 主に縮小するのでINTER_AREA使用
#                 testimage = cv2.resize(
#                     testimage, imagesize, interpolation=cv2.INTER_AREA)
                # チャンネルの次元がないので1次元追加する
                testimage = np.expand_dims(testimage, -1)
#                 testimage = np.asarray(testimage, dtype=np.float32).reshape(
#                     (imagesize[1], imagesize[0], 1))
                # チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
                #testimage = testimage.transpose(2, 0, 1)

            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, sorted(files)  # 画像リストとファイル名のリストを返す



# %% データ読み込み
# csvファイルを開く
def load_datasets():
    df_source = pd.read_csv('./datadirectory.csv')

    # 特定のヘッダを持つ列だけ取り出す
    df_train_image_dir = df_source['image_directory_train']
    df_train_label_dir = df_source['label_directory_train']

    df_test_image_dir = df_source['image_directory_test']
    df_test_label_dir = df_source['label_directory_test']


    # データ準備
    # training用画像読み込み
    train_image_org = np.empty(0)
    train_label_org = np.empty(0)
    for i in range(df_train_image_dir.count()):
        train_image_temp, train_image_filenames_temp = load_images(
            df_train_image_dir.iat[i], 'Color')
        train_label_temp, train_label_filenames_temp = load_images(
            df_train_label_dir.iat[i], 'Gray')
        if len(train_image_org) == 0:
            train_image_org = train_image_temp
            train_label_org = train_label_temp
        else:
            train_image_org = np.concatenate([train_image_org, train_image_temp], axis=0)
            train_label_org = np.concatenate([train_label_org, train_label_temp], axis=0)

    # test用画像読み込み
    test_image_org = np.empty(0)
    test_label_org = np.empty(0)
    test_filenames_org = []  # 結果書き出し用にファイル名リストを格納する
    for i in range(df_test_image_dir.count()):
        test_image_temp, test_image_filenames_temp = load_images(
            df_test_image_dir.iat[i], 'Color')
        test_label_temp, test_label_filenames_temp = load_images(
            df_test_label_dir.iat[i],  'Gray')
        test_filenames_org.append(test_image_filenames_temp)
        if len(test_image_org) == 0:
            test_image_org = test_image_temp
            test_label_org = test_label_temp
        else:
            test_image_org = np.concatenate([test_image_org, test_image_temp], axis=0)
            test_label_org = np.concatenate([test_label_org, test_label_temp], axis=0)

    test_filenames_org = np.asarray(test_filenames_org, dtype=object)
    test_filenames_org = test_filenames_org.reshape(-1)

    print('Data load finished')
    print('Data numbers for train: ' + repr(len(train_image_org)) +
          ', test: ' + repr(len(test_image_org)))
    return train_image_org, train_label_org, test_image_org, test_label_org, test_filenames_org

# train_image, test_image, train_label, test_label = train_test_split(all_image, all_label, test_size=0.2)#1つのデータセットをtrainとtest用に分割する場合


