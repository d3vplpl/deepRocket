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
from os import listdir
from os.path import isfile, join

CURRENT_YEAR = str(date.today().year)
path_mst = (os.path.join(os.path.curdir, 'mst'))
csv_files = (os.path.join(os.path.curdir, 'csv_files'))
train_files = (os.path.join(os.path.curdir, 'train_files'))

# check if directory exists, if not create it
if not (os.path.isdir('csv_files')):
    os.mkdir('csv_files')
if not (os.path.isdir('train_files')):
    os.mkdir('train_files')



fieldnames = ['label', 'price1', 'price2', 'price3', 'price4'] #  nagłówki pliku csv pliku train


def data_normalize(x, whole_set_of_Xs):
    #x - min
    #max- min
    x = x[0]
    min_x = min(whole_set_of_Xs)
    min_x = min_x[0]
    max_x = max(whole_set_of_Xs)
    max_x = max_x[0]
    if max_x - min_x != 0:
        z = (x - min_x) / (max_x - min_x)
    else:
        z = 0
    result = [z]
    return result


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


def process_bossa_data(cutoff):

    files = [f for f in (os.listdir((os.path.join(os.path.curdir, 'mst')))) if os.path.isfile(os.path.join(path_mst, f))]\

    for fi in files:
        print('Processing file: ', fi)
        f = open(os.path.join(path_mst, fi))
        cs = csv.reader(f)
        cs_list = list(cs)
        if len(cs_list) < 6 + cutoff:
            continue  # if there is too little data on given ticker then ignore it
        cs_list = cs_list[0: len(cs_list)-cutoff]  # ucinamy z końca tyle ile wynosi cutoff
        date_of_last = (cs_list[len(cs_list)-1][1])  # tutaj jest data ostaniego notowania
        print (date_of_last)
        #if(date_of_last[:4]) != CURRENT_YEAR:
        #    continue

        header = cs_list.pop(0)  # removes header
        if str(header).find('OPENINT') != -1:   # if openint is found then eliminate - we don't need these
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
            f_name = os.path.join(csv_files, ticker + '.csv')
            with open(f_name, mode='w') as csv_file:
                rocket_label_element = [1]
                plummet_label_element = [2]
                list_to_normalize = [float_list[-5], float_list[-4], float_list[-3], float_list[-2]]
                label = [0]
                if is_rocket:
                    label = rocket_label_element
                if is_plummet:
                    label = plummet_label_element

                n1, n2, n3, n4 = data_normalize(float_list[-5],list_to_normalize), data_normalize(float_list[-4],list_to_normalize), \
                                         data_normalize(float_list[-3],list_to_normalize), data_normalize(float_list[-2],list_to_normalize)
                list1 = (label + n1 + n2 + n3 + n4)

                print('list to be written: ', list1)

                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
                writer.writeheader()
                csv_wr = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                            lineterminator='\n')
                csv_wr.writerow(list1)
    print('Finished successfully for cutoff: ', cutoff)

img_rows, img_cols = 1, 4  #tu było 28, 28
num_classes = 3


def data_prep(raw):
    print('raw label is: ', raw.label)
    print('num_classes is: ', num_classes)
    num_images = raw.shape[0]
    print('num_images is: ', num_images)
    out_y = keras.utils.to_categorical(raw.label, num_classes, dtype='float32')

    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array
    #out_x = x_shaped_array / 255 #tego nie powinno być bo ma zastosowanie do kolorów
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

    train_file = "combined_train_18032020.csv"  # tutaj muszę dać mój plik z labelami
    raw_data = pd.read_csv(train_file)
    #print(raw_data)
    x, y = data_prep(raw_data)  # wymiary muszą być dobre
    my_model = Sequential()
    my_model.add(Conv2D(20, kernel_size=(1, 2),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))
    my_model.add(Conv2D(20, kernel_size=(1, 2), activation='relu'))
    my_model.add(Flatten())
    my_model.add(Dense(128, activation='relu'))
    my_model.add(Dense(num_classes, activation='softmax'))

    my_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    my_model.fit(x, y,
              batch_size=128,
              epochs=2,
              validation_split=0.2)


def merge_csv_files(path_csv):

    extension = 'csv'
    all_filenames = [f for f in listdir(path_csv) if isfile(join(path_csv, f))]
    print('Objects to concat', all_filenames)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(join(path_csv, f)) for f in all_filenames])
    # export to csv
    filename = 'combined_csv.csv'
    if path_csv == csv_files:
        filename = str(date.today().day) + str(date.today().month) + str(date.today().year) + '_cutoff' + str(cutoff) + '.csv'
    combined_csv.to_csv(filename, index=False, encoding='utf-8')

    if delete:
        for f1 in os.listdir(path_csv):
            print('Deleting: ', f1)
            if filename.endswith('.csv'):
                os.unlink(join(path_csv, f1))


delete = False
cutoff = 0
get_data()
process_bossa_data(cutoff)
merge_csv_files(csv_files)  # this merges small csv per ticket files into one, daily csv for all the tickets
merge_csv_files(train_files) # this merges daily csv files into train file


train_my_model()

