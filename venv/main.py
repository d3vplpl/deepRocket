__author__ = 'md'
import requests
import os
import zipfile
import csv
import numpy
import pandas as pd
import csv
from datetime import date
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

CURRENT_YEAR = str(date.today().year)
path_mst = (os.path.join(os.path.curdir, 'mst'))
path_csv = (os.path.join(os.path.curdir, 'csv_files'))


def get_data():

    url = 'http://bossa.pl/pub/metastock/mstock/mstall.zip'
    print('Getting http://bossa.pl/pub/metastock/mstock/mstall.zip...')
    req = requests.get(url)
    print('Success!')
    # check if directory exists, if not create it
    if not (os.path.isdir('file')):
        os.mkdir('file')
    stock_file = open(os.path.join('file', os.path.basename(url)), 'wb')
    for chunk in req.iter_content(100000):
        stock_file.write(chunk)
    stock_file.close()
    zipf = zipfile.ZipFile(os.path.join('file', os.path.basename(url)))
    zipf.extractall((os.path.join(os.path.curdir, 'mst')))


#get_data()

def process_bossa_data():

    files = [f for f in (os.listdir((os.path.join(os.path.curdir, 'mst')))) if os.path.isfile(os.path.join(path_mst, f))]\

    for fi in files:

        f = open(os.path.join(path_mst, fi))
        cs = csv.reader(f)
        cs_list = list(cs)
        date_of_last = (cs_list[len(cs_list)-1][1])
        if(date_of_last[:4]) != CURRENT_YEAR:
            continue

        header = cs_list.pop(0)  # removes header
        if str(header).find('OPENINT') != -1:   # if openint is found - we don't need these
            print('open int found ', fi, header)
            continue

        float_list = []
        ticker = ''

        for el in cs_list:
            ticker = el.pop(0)  # removes ticker
            removed_date = el.pop(0)  # removes date
            remove_1 = el.pop(0)
            remove_2 = el.pop(0)
            remove_3 = el.pop(0)
            remove_4 = el.pop(1)  # remove all the prices except 1 also remove the volume
            float_list.append([float(i) for i in el])
        n_array = numpy.array(float_list)

        closes = n_array[:, 0]  # every row of column 0, basically just take the column 0

        float_list = n_array.tolist()
        #print(closes)
        max_close = max(closes)
        min_close = min(closes)
        last_close = float_list[len(float_list)-1]
        #print(last_close)
        last_close = last_close[0]
        # print('Max close:', max_close)
        is_rocket = False
        is_plummet = False
        if last_close == max_close:
            is_rocket = True
            print(ticker, ' is a rocket')
        if last_close == min_close:
            is_plummet = True
            print(ticker, ' is a plummet')

        if is_rocket or is_plummet:
            f_name = os.path.join(path_csv, ticker + '.csv')
            with open(f_name, mode='w') as csv_file:
                rocket_label_element = [1]
                plummet_label_element = [2]
                try:
                    if is_rocket:
                        list1 = (rocket_label_element + float_list[-5] + (float_list[-4]) + float_list[-3] + (float_list[-2]) + float_list[-1])
                    if is_plummet:
                        list1 = (plummet_label_element + float_list[-5] + (float_list[-4]) + float_list[-3] + (
                        float_list[-2]) + float_list[-1])

                    print('list to be written: ', list1)
                    fieldnames = ['label', 'price1', 'price2', 'price3', 'price4', 'price5']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
                    writer.writeheader()
                    csv_wr = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                            lineterminator='\n')
                    csv_wr.writerow(list1)

                except:
                    print('Not enough prices for ticker ', ticker)


process_bossa_data()
img_rows, img_cols = 28, 28
num_classes = 10


def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


def train_model_template():

    train_file = "train.csv" #tutaj muszę dać mój plik z labelami
    raw_data = pd.read_csv(train_file)
    x, y = data_prep(raw_data)

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x, y,
              batch_size=128,
              epochs=2,
              validation_split=0.2)
#train_model_template()


def train_my_model():

    train_file = "train.csv"  # tutaj muszę dać mój plik z labelami
    raw_data = pd.read_csv(train_file)
    x, y = data_prep(raw_data) #wymiary muszą być dobre
    my_model = Sequential()
