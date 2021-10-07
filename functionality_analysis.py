from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score
from keras.losses import sparse_categorical_crossentropy
import config
from keras.models import load_model
import preprocess as pp


# 1. Functionality 

# 1.1 Accuracy (Classification report)

def model_classification_report(y_true,y_pred, labels):


    c_r = classification_report(y_true, y_pred,target_names=labels)
    
    print('[INFO] Classification Report: ')
    print(c_r)    
    return c_r



# 1.2 Confusion Matrix

def model_confusion_matrix(y_true,y_pred):


    c_m = confusion_matrix(y_true, y_pred)
    print("[INFO] Confusion matrix : ")
    print(c_m)


if __name__ == '__main__':

   model = load_model(config.model_path)
   #image_data = config.test_data
   size = model.input_shape[1:3]
   image_generator, image_class, image_label = pp.image_data_preprocess(config.test_data_path,size)
   predictions, predictions_idxs = pp.make_predictions(model, image_generator) 
   mcr = model_classification_report(image_class, predictions_idxs, image_label)
   cm =  model_confusion_matrix(image_class, predictions_idxs)
   
   
   




