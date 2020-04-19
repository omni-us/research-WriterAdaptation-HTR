import torch.utils.data as D
import cv2
import numpy as np
import marcalAugmentor

train_set = '/home/lkang/datasets/syn_60000_words/subset_20305words.txt' # 20305 words source domain
valid_set = '/home/lkang/datasets/iam_final_words/RWTH.iam_word_gt_final.valid.thresh'
test_set = '/home/lkang/datasets/iam_final_words/RWTH.iam_word_gt_final.test.thresh'
target_set = test_set

img_baseDir = dict()
img_baseDir['src'] = '/home/lkang/datasets/syn_60000_words/60000_words/'
img_baseDir['tar'] = '/home/lkang/datasets/iam_final_words/words/'

VGG_NORMAL = True

RM_BACKGROUND = True
FLIP = False # flip the image
OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_WIDTH = 1011 # m01-084-07-00 max_length
IMG_HEIGHT = 64

def filterOutShortImage(data_set_file, flag='src', thresh=64):
    new_file_name = data_set_file.split('/')[-1]+'.filter'
    with open(new_file_name, 'w') as out:
        with open(data_set_file, 'r') as sr_f:
            data = sr_f.readlines()
            ids = [i.split(' ')[0].split(',')[0] for i in data]
            for i in range(len(ids)):
                url = img_baseDir[flag] + ids[i] + '.png'
                img = cv2.imread(url, 0)
                rate = float(IMG_HEIGHT) / img.shape[0]
                img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
                width_new = img.shape[1]
                if width_new < thresh:
                    continue
                else:
                    out.write(data[i])
    print('Filter to file: '+new_file_name)
    return new_file_name

new_target_set = filterOutShortImage(target_set, 'tar', 32)

new_train_set = 'new_train_set.gt'
with open(new_train_set, 'w') as out:
    with open(train_set, 'r') as tr_f:
        tr_data = tr_f.readlines()
        for i in tr_data:
            out.write('src,'+i)
    with open(new_target_set, 'r') as te_f:
        te_data = te_f.read()
    out.write(te_data)

GT_TR = new_train_set
GT_VA = valid_set
GT_TE = test_set



# todo: allocate probabilities to each training item
def source_set_sample(source_list, num):
    return np.random.choice(source_list, num, replace=False)

def labelDictionary():
    labels = [' ', '!', '"', "'", '#', '&', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class IAM_words(D.Dataset):
    def __init__(self, file_label, trainset=True, augmentation=True):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.trainset = trainset
        self.augmentation = augmentation
        self.transformer = marcalAugmentor.augmentor

    def __getitem__(self, index):
        word = self.file_label[index]
        img, img_width, dom_label = self.readImage_keepRatio(word[0], flip=FLIP)
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens, dom_label)
        return word[0], img, img_width, label, dom_label

    def __len__(self):
        return len(self.file_label)

    # dom_label 0: real   1: synthetic
    def readImage_keepRatio(self, file_name, flip):
        ctx = file_name.split(',')
        if ctx[0] == 'src':
            dom_label = 0 # source domain
            img_flag = 'src'
        else:
            dom_label = 1 # target domain
            img_flag = 'tar'

        if RM_BACKGROUND:
            if dom_label == 0:
                file_name, thresh = ctx[1], ctx[2]
            elif dom_label == 1:
                file_name, thresh = ctx[0], ctx[1]

            thresh = int(thresh)
        url = img_baseDir[img_flag] + file_name + '.png'

        img = cv2.imread(url, 0)
        if RM_BACKGROUND:
            img[img>thresh] = 255

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        # c04-066-01-08.png 4*3, for too small images do not augment
        if self.augmentation: # augmentation for training data
            img_new = self.transformer(img)
            if img_new.shape[0] != 0 and img_new.shape[1] != 0:
                rate = float(IMG_HEIGHT) / img_new.shape[0]
                img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
            else:
                img = 255 - img
        else:
            img = 255 - img

        img_width = img.shape[-1]

        if flip: # because of using pack_padded_sequence, first flip, then pad it
            img = np.flip(img, 1)

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg[:, :img_width] = img
        outImg = outImg/255. #float64
        outImg = outImg.astype('float32')
        if VGG_NORMAL:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            outImgFinal = np.zeros([3, *outImg.shape])
            for i in range(3):
                outImgFinal[i] = (outImg - mean[i]) / std[i]
            return outImgFinal, img_width, dom_label

        else:
            outImg = np.vstack([np.expand_dims(outImg, 0)] * 3) # GRAY->RGB
            return outImg, img_width, dom_label

    def label_padding(self, labels, num_tokens, dom_label):
        new_label_len = []
        if self.trainset and dom_label: # target domain
            new_label_len.append(0)
            ll = []
            ll.extend([tokens['PAD_TOKEN']] * self.output_max_len)
        else: # source domain dom_label==0 and validation/test
            ll = [letter2index[i] for i in labels]
            num = self.output_max_len - len(ll) - 2
            new_label_len.append(len(ll)+2)
            ll = np.array(ll) + num_tokens
            ll = list(ll)
            ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
            if not num == 0:
                ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

def loadData():
    with open(GT_VA, 'r') as f_va:
        data_va = f_va.readlines()
        file_label_va = [i[:-1].split(' ') for i in data_va]

    with open(GT_TE, 'r') as f_te:
        data_te = f_te.readlines()
        file_label_te = [i[:-1].split(' ') for i in data_te]

    with open(GT_TR, 'r') as f_tr:
        data_tr = f_tr.readlines()
        file_label_tr = [i[:-1].split(' ') for i in data_tr]

    np.random.shuffle(file_label_tr)
    data_train = IAM_words(file_label_tr, trainset=True, augmentation=True)
    data_valid = IAM_words(file_label_va, trainset=False, augmentation=False)
    data_test = IAM_words(file_label_te, trainset=False, augmentation=False)
    return data_train, data_valid, data_test

if __name__ == '__main__':
    pass
