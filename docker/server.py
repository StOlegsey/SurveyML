import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel

model_v2 = load_model("model_v2.h5")
model_training_lvl = load_model("training_lvl_model.h5")
course_lvls = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])#уровни подготовки для курсов

app = FastAPI()

class InputData(BaseModel):
    data: list[int]

@app.post("/predict/")
async def predict(input_data: InputData):
    answers = input_data.data
    
    predictions = predict_with_course_lvl(model_v2, np.array([answers]))
    predictions = int(np.argmax(predictions, axis=1))

    return {"course": predictions}

def fit_course_lvl(answers, predicted_lvl):
        for j in range(len(answers)):
            for i in range(len(answers[j])):
                if predicted_lvl[j] < course_lvls[i]:
                    if answers[j][i] < 0:
                        answers[j][i] *= (course_lvls[i] - predicted_lvl[j] + 1)
                    else:
                        answers[j][i] /= (course_lvls[i] - predicted_lvl[j] + 1)
                    
def predict_with_course_lvl(model, answers_raw):
    #Передаем модель и результаты ответов
    answers_no_lvl = np.array([x[3:] for x in answers_raw])
    answers_training_lvl = np.array([x[0:3] for x in answers_raw])
    train_lvl_pred = model_training_lvl.predict(answers_training_lvl)#предполагаемый уровень знаний основываясь на ответах
    train_lvl_pred = np.argmax(train_lvl_pred, axis=1)#окргуляем предполагаемый уровень знаний основываясь на ответах
    final_pred = model.predict(answers_no_lvl)#предпалагаем курс, не учитывая первые 3 вопроса
    #print('raw predictions\n', predictions_no_lvl)
    fit_course_lvl(final_pred, train_lvl_pred)#уменьшаем вероятность для курсов, дял которых недостаточный уровень знаний
    return final_pred