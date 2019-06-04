# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_model_results(trained_model, file_path='output', model_name='resume_parser', result_type=('acc', 'val_acc')):
    """
    Plot the accuracy/loss of the trained model
    :param trained_model: trained keras model object
    :param file_path: output directory
    :param model_name: name of the model
    :param result_type: list of accuracy and loss
    :return:
    """
    plt.style.use('ggplot')
    plt.plot(trained_model.history[result_type[0]])
    plt.plot(trained_model.history[result_type[1]])
    if result_type[0] == 'acc':
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
    elif result_type[0] == 'loss':
        plt.title('Model Loss')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')

    plt.legend(['Train', 'Validation'], loc='upper left')

    # save the plot
    plt.tight_layout()
    plt.savefig("{}/{}_{}.png".format(file_path, model_name, result_type[0]))
    plt.close()