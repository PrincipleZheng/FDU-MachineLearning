import matplotlib.pyplot as plt

def show(x_labels, y_labels, *arg):
    if len(arg) != 0:
        rates = arg[0]
        for i in range(len(x_labels)):
            x_label = x_labels[i]
            y_label = y_labels[i]
            rate = rates[i]
            plt.plot(x_label, y_label, label='split rate='+str(rate))
            plt.scatter(x_label, y_label)
        # for i in range(len(x_label)):
        #     if i % 2 == 0:
        #         plt.annotate(y_label[i], xy=(x_label[i], y_label[i]), xytext=(x_label[i], y_label[i]+0.001), weight='light')
        #     else:
        #         plt.annotate(y_label[i], xy=(x_label[i], y_label[i]), xytext=(x_label[i], y_label[i]-0.003), weight='light')
    else:
        plt.plot(x_labels, y_labels, label='KFold')
        plt.scatter(x_labels, y_labels)
    plt.legend(loc='best')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuracy of iris dataset using KNN algorithm with differenct k value')
    plt.show()