from src.classifier.context_lstm_cnn import ContextLstmCnn
from src.util.text_bundler import bundle_text_label_pairs
from src.util.text_fitter_util import fit_text


class SentenceClassifier(object):
    def __init__(self):
        self.classifier = ContextLstmCnn()


    def load_model(self, model_dir_path):
        self.classifier.load_model(model_dir_path)


    def fit(self, training_data_dir_path, model_dir_path, batch_size=None, epochs=None,
                            test_size=None, random_state=None):
        # since type and label of sentence is processed in one go.
        text_data_model = fit_text(training_data_dir_path)
        text_label_pairs = bundle_text_label_pairs(training_data_dir_path)

        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 30

        history = self.classifier.fit(text_data_model=text_data_model,
                                      model_dir_path=model_dir_path,
                                      text_type_label_tuples=text_label_pairs,
                                      batch_size=batch_size, epochs=epochs,
                                      test_size=test_size,
                                      random_state=random_state)
        return history


    def predict_resume_context(self, texts):
        education, experience, project, skills, personal = list(), list(), list(), list(), list()
        for index, sent in enumerate(texts.splitlines()):
            labels = self.classifier.predict(sent)

            if labels[1] == 'education' or labels[1] == 'education_main':
                education.append((index, labels[0], labels[1], sent))
            elif labels[1] == 'experience' or labels[1] == 'experience_main':
                experience.append((index, labels[0], labels[1], sent))
            elif labels[1] == 'project' or labels[1] == 'project_main':
                project.append((index, labels[0], labels[1], sent))
            elif labels[1] == 'skill':
                skills.append((index, labels[0], labels[1], sent))
            elif labels[1] == 'personal':
                personal.append((index, labels[0], labels[1], sent))

        all = dict()
        all['education'] = education
        all['experience'] = experience
        all['project'] = project
        all['skill'] = skills
        all['personal'] = personal

        return all
