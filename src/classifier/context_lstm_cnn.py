import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import model_from_json
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from src.util.plot import plot_model_results
from src.util.text_tokenizer import word_tokenizer


class ContextLstmCnn(object):
    model_name = 'content-lstm-cnn'

    def __init__(self):
        self. model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.vocab_size = None
        self.line_types = None
        self.line_labels = None


    @staticmethod
    def get_architecture_file_path(model_dir_path):
        """
        :param model_dir_path: path of directory where model is stored
        :return: It returns the architecture of LSTM model that is saved in json format in predifined path.
        """
        return model_dir_path + '/' + ContextLstmCnn.model_name + '_architecture.json'


    @staticmethod
    def get_weight_file_path(model_dir_path):
        """
        :param model_dir_path: path of directory where model is stored
        :return: It returns the file where model will be saved.
        """
        return model_dir_path + '/' + ContextLstmCnn.model_name + '_weights.h5'


    @staticmethod
    def get_config_file_path(model_dir_path):
        """
        :param model_dir_path: path of directory where model is stored
        :return: returns the file for storing the config data. ie; word2idx, idx2word, max_len, etc
        """
        return model_dir_path + '/' + ContextLstmCnn.model_name + '_config.npy'


    def create_model(self):
        print('model')

    def load_model(self, model_dir_path):
        json = open(self.get_architecture_file_path(model_dir_path), 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        config_file_path = self.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).items()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.line_types = self.config['line_types']
        self.line_labels = self.config['line_labels']

        print('Line Types : ', self.line_types)
        print('Line Labels : ', self.line_labels)


    def generate_input_mappings(self, text_type_label_tuples):
        xs = []
        y_type, y_label = [], []

        for text_type, text_label, sentence in text_type_label_tuples:
            tokens = [x.lower() for x in word_tokenizer(sentence)]
            wid_list = list()

            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            # getting index of text_type(string) in line_types (dict)
            y_type.append(self.line_types[text_type])
            # getting index of text_label(string) in line_labels (dict)
            y_label.append(self.line_labels[text_label])

        X = pad_sequences(xs, self.max_len)
        Y_Type = np.array(y_type, dtype=np.float32)
        Y_Label = np.array(y_label, dtype=np.float32)

        Y_Type_one_hot_encoded = to_categorical(Y_Type)
        Y_Label_one_hot_encoded = to_categorical(Y_Label)

        print("X Padding at 0 : ", X[0])
        print("Y Type One Hot Encoding at 0 : ", Y_Type_one_hot_encoded[0])
        print("Y Label One Hot Encoding at 0 : ", Y_Label_one_hot_encoded[0])

        return X, Y_Type_one_hot_encoded, Y_Label_one_hot_encoded


    def fit(self, text_data_model, model_dir_path, text_type_label_tuples, batch_size, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 30
        if test_size is None:
            test_size = 0.3
        if random_state is None:
            random_state = 42

        self.config = text_data_model
        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.line_types = self.config['line_types']
        self.line_labels = self.config['line_labels']

        print('Line Labels : ',self.line_labels)
        print('Line Types : ', self.line_types)

        # saving config definition in file numpy
        np.save(self.get_config_file_path(model_dir_path), self.config)

        self.create_model()

        # saving model definition in json file
        json = self.model.to_json()
        open(self.get_architecture_file_path(model_dir_path), 'w').write(json)

        # converting words into the index number of 5000 most common words and labels into index number
        padded_sent, y_type_one_hot, y_label_one_hot = self.generate_input_mappings(text_type_label_tuples)

        split = train_test_split(padded_sent, y_type_one_hot, y_label_one_hot, test_size=test_size, random_state=random_state)
        (x_train, x_test, y_line_type_train, y_line_type_test, y_line_label_train, y_line_label_test) = split

        print("Input / Output data shapes for Line Type")
        print(x_train.shape, x_test.shape, y_line_type_train.shape, y_line_type_test.shape)

        print("Input / Output data shapes for Line Label")
        print(x_train.shape, x_test.shape, y_line_label_train.shape, y_line_label_test.shape)

        losses = {
            "line_type_output":"binary_crossentropy",
            "line_label_output":"categorical_crossentropy"
        }

        weight_file_path = self.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        classifier = self.model.fit(x_train,
                                    {"line_type_output": y_line_type_train, "line_label_output":y_line_label_train},
                                    validation_data=
                                    (x_test, {"line_type_output": y_line_type_test, "line_label_output": y_line_label_test}),
                                    batch_size=batch_size, epochs=epochs,
                                    callbacks=[checkpoint], verbose=1
                                    )

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + ContextLstmCnn.model_name + '_history.npy', classifier.history)

        # plot model diagram, accuracy and loss
        plot_model(self.model, show_shapes=True, to_file=model_dir_path + "/model.png")
        plot_model_results(classifier, model_dir_path, 'resume_parser_model',
                           ('line_type_output_acc', 'val_line_type_output_acc'))
        plot_model_results(classifier, model_dir_path, 'resume_parser_model',
                           ('line_type_output_loss', 'val_line_type_output_loss'))
        plot_model_results(classifier, model_dir_path, 'resume_parser_model',
                           ('line_label_output_acc', 'val_line_label_output_acc'))
        plot_model_results(classifier, model_dir_path, 'resume_parser_model',
                           ('line_label_output_loss', 'val_line_label_output_loss'))

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score : ', score[0])
        print('accuracy : ', score[1])

        return classifier


    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenizer(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 0 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)

        # get the predictions for line label and line type
        (line_type_proba, line_label_proba) = model.predict(x)
        line_type_idx = line_type_proba[0].argmax()  # index of line type with highest probability
        line_label_idx = line_label_proba[0].argmax()  # index of line label with highest probability

        idx2_line_label = dict([(idx, label) for label, idx in line_labels.items()])
        # print(idx2_line_label)
        idx2_line_type = dict([(idx, label) for label, idx in line_type_labels.items()])
        line_label = idx2_line_label[line_label_idx]
        line_type = idx2_line_type[line_type_idx]
        return line_label, line_type




